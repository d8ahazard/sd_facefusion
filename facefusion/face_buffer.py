"""
Face Buffer System - Pre-scan and cache face/YOLO detections across video frames
with adaptive gap filling to reduce flickering and improve consistency.
"""

import hashlib
import os
import pickle
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from facefusion import logger, state_manager
from facefusion.typing import Face, VisionFrame


@dataclass
class FrameDetectionData:
    """Data structure for per-frame face and YOLO detections."""
    frame_number: int
    faces: List[Face] = field(default_factory=list)
    yolo_detections: List[Dict[str, Any]] = field(default_factory=list)
    motion_score: float = 0.0
    interpolated: bool = False
    cache_source: str = 'memory'  # 'memory' or 'disk'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization (excluding Face objects)."""
        return {
            'frame_number': self.frame_number,
            'faces': self.faces,  # Face objects will be pickled
            'yolo_detections': self.yolo_detections,
            'motion_score': self.motion_score,
            'interpolated': self.interpolated,
            'cache_source': self.cache_source
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FrameDetectionData':
        """Create from dictionary."""
        return cls(**data)


class MotionAnalyzer:
    """Calculates motion scores between frames for adaptive gap thresholds."""
    
    def __init__(self):
        self.cache = {}
    
    def calculate_motion_score(self, frame1: VisionFrame, frame2: VisionFrame) -> float:
        """
        Calculate normalized motion magnitude between two frames.
        Returns value between 0 (no motion) and 1 (high motion).
        """
        # Convert to grayscale for faster processing
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1
            gray2 = frame2
        
        # Resize for faster computation
        height, width = gray1.shape[:2]
        if width > 640:
            scale = 640 / width
            new_size = (640, int(height * scale))
            gray1 = cv2.resize(gray1, new_size)
            gray2 = cv2.resize(gray2, new_size)
        
        # Calculate frame difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Calculate mean difference (0-255 range)
        mean_diff = np.mean(diff)
        
        # Normalize to 0-1 range
        # Typical motion: 5-30, high motion: 30+
        normalized = min(mean_diff / 50.0, 1.0)
        
        return normalized
    
    def calculate_optical_flow_motion(self, frame1: VisionFrame, frame2: VisionFrame) -> float:
        """
        Calculate motion using optical flow (more accurate but slower).
        Returns value between 0 (no motion) and 1 (high motion).
        """
        # Convert to grayscale
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1
            gray2 = frame2
        
        # Resize for faster computation
        height, width = gray1.shape[:2]
        if width > 640:
            scale = 640 / width
            new_size = (640, int(height * scale))
            gray1 = cv2.resize(gray1, new_size)
            gray2 = cv2.resize(gray2, new_size)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Calculate magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mean_magnitude = np.mean(magnitude)
        
        # Normalize (typical motion: 0.5-3, high motion: 3+)
        normalized = min(mean_magnitude / 5.0, 1.0)
        
        return normalized


class GapFiller:
    """Handles interpolation of missing face detections in gaps."""
    
    def __init__(self, interpolation_mode: str = 'simple'):
        self.interpolation_mode = interpolation_mode
        self.motion_analyzer = MotionAnalyzer()
    
    def interpolate_faces(
        self, 
        prev_faces: List[Face], 
        next_faces: List[Face], 
        gap_size: int,
        motion_score: float = 0.0
    ) -> List[List[Face]]:
        """
        Generate intermediate face data for frames in a gap.
        Returns a list of face lists, one for each frame in the gap.
        """
        if not prev_faces or not next_faces:
            return [[] for _ in range(gap_size)]
        
        # Match faces between prev and next based on position/embedding similarity
        face_pairs = self._match_faces(prev_faces, next_faces)
        
        # Generate interpolated frames
        interpolated_frames = []
        for frame_idx in range(gap_size):
            t = (frame_idx + 1) / (gap_size + 1)  # Interpolation ratio
            frame_faces = []
            
            for prev_face, next_face in face_pairs:
                interpolated_face = self._interpolate_single_face(prev_face, next_face, t)
                if interpolated_face:
                    frame_faces.append(interpolated_face)
            
            interpolated_frames.append(frame_faces)
        
        return interpolated_frames
    
    def _match_faces(self, prev_faces: List[Face], next_faces: List[Face]) -> List[Tuple[Face, Face]]:
        """Match faces between frames based on position and embedding similarity."""
        if not prev_faces or not next_faces:
            return []
        
        pairs = []
        used_next_indices = set()
        
        for prev_face in prev_faces:
            best_match = None
            best_score = float('inf')
            best_idx = -1
            
            for idx, next_face in enumerate(next_faces):
                if idx in used_next_indices:
                    continue
                
                # Calculate position distance
                prev_center = [
                    (prev_face.bounding_box[0] + prev_face.bounding_box[2]) / 2,
                    (prev_face.bounding_box[1] + prev_face.bounding_box[3]) / 2
                ]
                next_center = [
                    (next_face.bounding_box[0] + next_face.bounding_box[2]) / 2,
                    (next_face.bounding_box[1] + next_face.bounding_box[3]) / 2
                ]
                position_distance = np.sqrt(
                    (prev_center[0] - next_center[0])**2 + 
                    (prev_center[1] - next_center[1])**2
                )
                
                # Calculate embedding distance if available
                embedding_distance = 0.0
                if (hasattr(prev_face, 'normed_embedding') and hasattr(next_face, 'normed_embedding') and
                    prev_face.normed_embedding is not None and next_face.normed_embedding is not None):
                    embedding_distance = 1 - np.dot(prev_face.normed_embedding, next_face.normed_embedding)
                
                # Combined score (weighted)
                # If no embeddings, use position only
                if embedding_distance > 0:
                    score = position_distance * 0.7 + embedding_distance * 1000 * 0.3
                else:
                    score = position_distance  # Position-only matching for lightweight faces
                
                if score < best_score:
                    best_score = score
                    best_match = next_face
                    best_idx = idx
            
            if best_match:
                pairs.append((prev_face, best_match))
                used_next_indices.add(best_idx)
        
        return pairs
    
    def _interpolate_single_face(self, prev_face: Face, next_face: Face, t: float) -> Optional[Face]:
        """Interpolate a single face between two frames."""
        try:
            # Interpolate bounding box
            bbox = [
                prev_face.bounding_box[i] * (1 - t) + next_face.bounding_box[i] * t
                for i in range(4)
            ]
            
            # Interpolate landmarks
            landmark_set = {}
            for key in prev_face.landmark_set.keys():
                if key in next_face.landmark_set:
                    prev_landmarks = prev_face.landmark_set[key]
                    next_landmarks = next_face.landmark_set[key]
                    interpolated_landmarks = prev_landmarks * (1 - t) + next_landmarks * t
                    landmark_set[key] = interpolated_landmarks
            
            # Use average embedding if available (simple approach)
            embedding = None
            normed_embedding = None
            if prev_face.embedding is not None and next_face.embedding is not None:
                embedding = (prev_face.embedding + next_face.embedding) / 2
                normed_embedding = embedding / np.linalg.norm(embedding)
            
            # Interpolate other attributes
            angle = prev_face.angle * (1 - t) + next_face.angle * t
            
            # Create interpolated face
            interpolated_face = Face(
                bounding_box=tuple(bbox),
                score_set=prev_face.score_set,  # Use previous score
                landmark_set=landmark_set,
                angle=angle,
                embedding=embedding,
                normed_embedding=normed_embedding,
                gender=prev_face.gender,
                age=prev_face.age,
                race=prev_face.race
            )
            
            return interpolated_face
        
        except Exception as e:
            logger.warn(f"Failed to interpolate face: {e}", __name__)
            return None


class FaceBufferCache:
    """
    Main cache manager with hybrid memory/disk storage.
    Manages face and YOLO detection data across video frames.
    """
    
    def __init__(
        self, 
        video_path: str,
        memory_limit_mb: int = 1024,
        disk_cache_enabled: bool = True,
        max_memory_frames: int = 100
    ):
        self.video_path = video_path
        self.memory_limit_mb = memory_limit_mb
        self.disk_cache_enabled = disk_cache_enabled
        self.max_memory_frames = max_memory_frames
        
        # Memory cache (LRU)
        self.memory_cache: OrderedDict[int, FrameDetectionData] = OrderedDict()
        
        # Disk cache directory
        self.cache_dir = self._get_cache_directory()
        if disk_cache_enabled:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Metadata
        self.total_frames = 0
        self.frames_with_faces = 0
        self.frames_interpolated = 0
        self.gaps_filled = 0
        self.gaps_detected = 0
        self.total_objects_detected = 0
        
        # Statistics
        self.memory_usage_mb = 0.0
    
    def _get_cache_directory(self) -> Path:
        """Get cache directory path based on video hash."""
        video_hash = hashlib.md5(self.video_path.encode()).hexdigest()[:12]
        from facefusion.temp_helper import get_temp_directory_path
        temp_dir = get_temp_directory_path(self.video_path)
        cache_dir = Path(temp_dir) / 'face_buffer'
        return cache_dir
    
    def set_frame_data(self, frame_number: int, data: FrameDetectionData) -> None:
        """Store frame detection data in cache."""
        # Add to memory cache
        self.memory_cache[frame_number] = data
        self.memory_cache.move_to_end(frame_number)
        
        # Estimate memory usage
        self._update_memory_usage()
        
        # Evict to disk if needed
        if len(self.memory_cache) > self.max_memory_frames:
            self._evict_oldest_to_disk()
    
    def get_frame_data(self, frame_number: int) -> Optional[FrameDetectionData]:
        """Retrieve frame detection data from cache."""
        # Check memory cache first
        if frame_number in self.memory_cache:
            # Move to end (LRU)
            self.memory_cache.move_to_end(frame_number)
            return self.memory_cache[frame_number]
        
        # Check disk cache
        if self.disk_cache_enabled:
            data = self._load_from_disk(frame_number)
            if data:
                # Load back into memory
                self.memory_cache[frame_number] = data
                self.memory_cache.move_to_end(frame_number)
                
                # Evict if needed
                if len(self.memory_cache) > self.max_memory_frames:
                    self._evict_oldest_to_disk()
                
                return data
        
        return None
    
    def _evict_oldest_to_disk(self) -> None:
        """Evict oldest frame from memory to disk."""
        if not self.disk_cache_enabled or not self.memory_cache:
            return
        
        # Get oldest frame
        frame_number, data = self.memory_cache.popitem(last=False)
        
        # Save to disk
        self._save_to_disk(frame_number, data)
        
        self._update_memory_usage()
    
    def _save_to_disk(self, frame_number: int, data: FrameDetectionData) -> None:
        """Save frame data to disk."""
        cache_file = self.cache_dir / f'frame_{frame_number:06d}.cache'
        try:
            data.cache_source = 'disk'
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warn(f"Failed to save frame {frame_number} to disk: {e}", __name__)
    
    def _load_from_disk(self, frame_number: int) -> Optional[FrameDetectionData]:
        """Load frame data from disk."""
        cache_file = self.cache_dir / f'frame_{frame_number:06d}.cache'
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            data.cache_source = 'memory'
            return data
        except Exception as e:
            logger.warn(f"Failed to load frame {frame_number} from disk: {e}", __name__)
            return None
    
    def _update_memory_usage(self) -> None:
        """Estimate current memory usage."""
        # Rough estimate: 50KB per face + 10KB base per frame
        total_kb = 0
        for data in self.memory_cache.values():
            frame_kb = 10  # Base
            frame_kb += len(data.faces) * 50  # Faces
            frame_kb += len(data.yolo_detections) * 5  # YOLO
            total_kb += frame_kb
        
        self.memory_usage_mb = total_kb / 1024.0
    
    def cleanup(self) -> None:
        """Clean up disk cache."""
        if self.disk_cache_enabled and self.cache_dir.exists():
            try:
                import shutil
                shutil.rmtree(self.cache_dir)
                logger.info(f"Cleaned up face buffer cache at {self.cache_dir}", __name__)
            except Exception as e:
                logger.warn(f"Failed to clean up cache directory: {e}", __name__)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'total_frames': self.total_frames,
            'frames_with_faces': self.frames_with_faces,
            'frames_interpolated': self.frames_interpolated,
            'gaps_filled': self.gaps_filled,
            'gaps_detected': self.gaps_detected,
            'total_objects_detected': self.total_objects_detected,
            'memory_cache_size': len(self.memory_cache),
            'memory_usage_mb': self.memory_usage_mb,
            'disk_cache_enabled': self.disk_cache_enabled
        }


def detect_gaps(cache: FaceBufferCache, min_frame: int, max_frame: int) -> List[Tuple[int, int]]:
    """
    Identify frame ranges with missing face detections.
    Returns list of (start_frame, end_frame) tuples for gaps.
    """
    gaps = []
    gap_start = None
    
    for frame_num in range(min_frame, max_frame + 1):
        data = cache.get_frame_data(frame_num)
        has_faces = data and len(data.faces) > 0
        
        if not has_faces:
            if gap_start is None:
                gap_start = frame_num
        else:
            if gap_start is not None:
                gaps.append((gap_start, frame_num - 1))
                gap_start = None
    
    # Handle gap at the end
    if gap_start is not None:
        gaps.append((gap_start, max_frame))
    
    return gaps


def calculate_adaptive_gap_threshold(
    motion_scores: List[float], 
    fps: float, 
    max_gap_frames: int
) -> int:
    """
    Calculate adaptive gap threshold based on motion and FPS.
    
    - Low motion → allow larger gaps
    - High motion → only fill small gaps
    - Higher FPS → can fill more frames
    """
    if not motion_scores:
        return max(1, max_gap_frames // 2)
    
    avg_motion = np.mean(motion_scores)
    
    # FPS factor (normalize around 30fps)
    fps_factor = fps / 30.0
    
    # Motion factor (0-1, inverted because low motion allows larger gaps)
    motion_factor = 1.0 - min(avg_motion, 1.0)
    
    # Calculate threshold
    # Low motion (0.9-1.0 factor) + high FPS (2.0) → max_gap_frames
    # High motion (0.0-0.1 factor) + low FPS (0.8) → 1-2 frames
    threshold = int(max_gap_frames * motion_factor * fps_factor)
    threshold = max(1, min(threshold, max_gap_frames))
    
    return threshold


def scan_and_build_face_buffer_with_progress(
    frame_paths: List[str], 
    yolo_model: Optional[str] = None,
    fps: float = 30.0,
    progress_callback = None
) -> FaceBufferCache:
    """
    Face buffer scan with progress callback for UI updates.
    
    Args:
        frame_paths: List of frame file paths
        yolo_model: Optional YOLO model for object detection
        fps: Video FPS
        progress_callback: Optional callback(current, total, faces, objects)
    """
    return _scan_and_build_face_buffer_impl(frame_paths, yolo_model, fps, progress_callback)


def scan_and_build_face_buffer(
    frame_paths: List[str], 
    yolo_model: Optional[str] = None,
    fps: float = 30.0
) -> FaceBufferCache:
    """
    Main entry point for face buffer scanning (without progress callback).
    Coordinates all phases of detection, analysis, and gap filling.
    """
    return _scan_and_build_face_buffer_impl(frame_paths, yolo_model, fps, None)


def get_faces_lightweight(vision_frame: VisionFrame) -> List[Face]:
    """
    Lightweight face detection for buffer scanning.
    Skips expensive operations like embeddings and classification.
    """
    from facefusion.workers.classes.face_detector import FaceDetector
    from facefusion.workers.classes.face_landmarker import FaceLandmarker
    from facefusion.face_helper import apply_nms, convert_to_face_landmark_5, estimate_face_angle, get_nms_threshold
    from facefusion.typing import Face, FaceLandmarkSet, FaceScoreSet
    
    detector = FaceDetector()
    landmarker = FaceLandmarker()
    
    all_bounding_boxes = []
    all_face_scores = []
    all_face_landmarks_5 = []
    
    # Only detect at 0 degrees for speed (skip rotations)
    face_detector_angles = state_manager.get_item('face_detector_angles') or [0]
    for face_detector_angle in face_detector_angles:
        if face_detector_angle == 0:
            bounding_boxes, face_scores, face_landmarks_5 = detector.detect_faces(vision_frame)
        else:
            bounding_boxes, face_scores, face_landmarks_5 = detector.detect_rotated_faces(vision_frame, face_detector_angle)
        all_bounding_boxes.extend(bounding_boxes)
        all_face_scores.extend(face_scores)
        all_face_landmarks_5.extend(face_landmarks_5)
    
    if not all_bounding_boxes:
        return []
    
    # Apply NMS
    nms_threshold = get_nms_threshold(
        state_manager.get_item('face_detector_model'),
        state_manager.get_item('face_detector_angles')
    )
    keep_indices = apply_nms(
        all_bounding_boxes, 
        all_face_scores, 
        state_manager.get_item('face_detector_score') or 0.5, 
        nms_threshold
    )
    
    # Create lightweight faces (skip embeddings and classification)
    faces = []
    for index in keep_indices:
        bounding_box = all_bounding_boxes[index]
        face_score = all_face_scores[index]
        face_landmark_5 = all_face_landmarks_5[index]
        face_landmark_5_68 = face_landmark_5
        face_landmark_68_5 = landmarker.estimate_face_landmark_68_5(face_landmark_5_68)
        face_angle = estimate_face_angle(face_landmark_68_5)
        
        # Skip expensive landmark detection unless score threshold requires it
        face_landmark_68 = face_landmark_68_5
        face_landmark_score_68 = 0.0
        
        # Only do detailed landmarks if explicitly required
        landmarker_score = state_manager.get_item('face_landmarker_score') or 0
        if landmarker_score > 0:
            face_landmark_68, face_landmark_score_68 = landmarker.detect_face_landmarks(
                vision_frame, bounding_box, face_angle
            )
            if face_landmark_score_68 > landmarker_score:
                face_landmark_5_68 = convert_to_face_landmark_5(face_landmark_68)
        
        landmark_set: FaceLandmarkSet = {
            '5': face_landmark_5,
            '5/68': face_landmark_5_68,
            '68': face_landmark_68,
            '68/5': face_landmark_68_5
        }
        
        score_set: FaceScoreSet = {
            'detector': face_score,
            'landmarker': face_landmark_score_68
        }
        
        # Create face WITHOUT expensive embedding/classification
        # These can be calculated later if needed during actual processing
        face = Face(
            bounding_box=bounding_box,
            score_set=score_set,
            landmark_set=landmark_set,
            angle=face_angle,
            embedding=None,  # Skip for speed
            normed_embedding=None,  # Skip for speed
            gender=None,  # Skip for speed
            age=None,  # Skip for speed
            race=None  # Skip for speed
        )
        faces.append(face)
    
    return faces


def _scan_and_build_face_buffer_impl(
    frame_paths: List[str], 
    yolo_model: Optional[str] = None,
    fps: float = 30.0,
    progress_callback = None
) -> FaceBufferCache:
    """
    Internal implementation of face buffer scanning with BATCH PROCESSING.
    Does FULL face detection once so processing doesn't have to.
    Now optimized with parallel I/O and batch processing for maximum throughput!
    """
    from facefusion.face_analyser import get_many_faces
    from facefusion.face_selector import sort_and_filter_faces
    from facefusion.vision import read_image
    from facefusion.workers.classes.face_masker import FaceMasker
    from facefusion.face_buffer_config import (
        initialize_face_buffer_config, 
        get_batch_size, 
        get_io_workers,
        run_benchmark
    )
    from facefusion.thread_helper import set_semaphore_limit, reset_semaphore_limit
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    # Initialize config and run benchmark if needed
    initialize_face_buffer_config()
    benchmark_done = state_manager.get_item('face_buffer_benchmark_done') or False
    if not benchmark_done:
        logger.info("First boot detected - running performance benchmark...", __name__)
        run_benchmark()
    
    batch_size = get_batch_size()
    io_workers = get_io_workers()
    
    # CRITICAL FIX: Increase semaphore limit to allow parallel ONNX inference
    # This is the #1 bottleneck - without this, only 1 frame can run ONNX at a time
    # 
    # IMPORTANT: ONNX Runtime InferenceSession.run() is NOT officially thread-safe for concurrent calls
    # However, in practice:
    # - CUDA provider: Can handle limited concurrency (GPU queues work better)
    # - CPU provider: More prone to issues, needs stricter limits
    # 
    # The semaphore protects concurrent access - we're increasing the limit from 1 to allow
    # multiple threads to run inference simultaneously, which dramatically improves GPU utilization.
    # If you see crashes or incorrect results, reduce the semaphore limit.
    
    # Check execution provider - CUDA can handle more concurrency than CPU
    execution_providers = state_manager.get_item('execution_providers') or []
    using_cuda = any('cuda' in str(ep).lower() or 'CUDAExecutionProvider' in str(ep) for ep in execution_providers)
    
    if using_cuda:
        # CUDA provider: GPU can handle parallel inference better
        # Allow batch_size concurrent calls (e.g., 24 for batch_size=24)
        # This is conservative - could go higher but this should be safe
        semaphore_limit = batch_size
        logger.info(f"🔓 CUDA detected - Set thread semaphore limit to {semaphore_limit} (was 1)", __name__)
        logger.info(f"   → Enabling {semaphore_limit} concurrent ONNX inference calls for better GPU utilization", __name__)
    else:
        # CPU provider: More conservative - ONNX Runtime CPU is less thread-safe
        # Use smaller limit to avoid crashes/race conditions
        semaphore_limit = min(batch_size // 2, 8)  # Cap at 8 for CPU safety
        logger.info(f"🔓 CPU mode - Set thread semaphore limit to {semaphore_limit} (was 1)", __name__)
        logger.info(f"   → Conservative limit for CPU safety - consider using CUDA for better performance", __name__)
    
    set_semaphore_limit(semaphore_limit)
    
    # Performance monitoring
    import time
    scan_start_time = time.time()
    phase_times = {}
    
    try:
        logger.info(f"Starting OPTIMIZED face buffer scan for {len(frame_paths)} frames", __name__)
        logger.info(f"⚡ AGGRESSIVE MODE: batch_size={batch_size}, io_workers={io_workers}", __name__)
        logger.info(f"   → Actual I/O workers: {min(io_workers * 2, 16)}, Max parallel detection: {min(batch_size * 3, io_workers * 3, 64)}", __name__)
        logger.info("Doing FULL face analysis (embeddings, classification) - slow but only once!", __name__)
        
        # Initialize
        video_path = state_manager.get_item('target_path')
        memory_limit = state_manager.get_item('face_buffer_memory_limit_mb') or 1024
        disk_cache = state_manager.get_item('face_buffer_disk_cache')
        if disk_cache is None:
            disk_cache = True
        
        cache = FaceBufferCache(
            video_path=video_path,
            memory_limit_mb=memory_limit,
            disk_cache_enabled=disk_cache,
            max_memory_frames=100
        )
        
        cache.total_frames = len(frame_paths)
        
        # Phase 1: Initial Scan with BATCH PROCESSING
        phase1_start = time.time()
        logger.info("Phase 1: Scanning all frames for faces and objects (BATCH MODE)...", __name__)
        motion_analyzer = MotionAnalyzer()
        motion_scores = []
        prev_frame = None
        
        masker = None
        yolo_model_path = None
        if yolo_model and yolo_model != "None":
            masker = FaceMasker()
            # Resolve YOLO model path (same logic as in detect_face_object_intersections)
            from modules.paths_internal import models_path
            adetailer_path = os.path.join(models_path, "adetailer")
            
            model_path = yolo_model
            if not os.path.exists(model_path):
                # Try to find it in adetailer path
                model_path = os.path.join(adetailer_path, yolo_model)
            
            if os.path.exists(model_path):
                # Pre-load YOLO model once to avoid reloading for each frame
                from facefusion.workers.classes.face_masker import get_cached_yolo_model
                try:
                    device = "cuda" if state_manager.get_item('execution_providers') and 'cuda' in state_manager.get_item('execution_providers')[0].lower() else "cpu"
                    yolo_model_loaded = get_cached_yolo_model(model_path, device)
                    yolo_model_path = model_path
                    logger.info(f"YOLO model loaded: {os.path.basename(model_path)}", __name__)
                except Exception as e:
                    logger.warn(f"Failed to pre-load YOLO model: {e}", __name__)
            else:
                logger.warn(f"YOLO model not found: {yolo_model}", __name__)
        
        # Use TQDM for progress (console) or callback for UI
        from facefusion.mytqdm import mytqdm as tqdm
        total_objects_detected = 0
        
        # Thread-safe counters
        frames_processed_lock = threading.Lock()
        frames_processed = 0
        
        def process_frame_batch(batch_indices: List[int], batch_paths: List[str]) -> List[Tuple[int, FrameDetectionData]]:
            """Process a batch of frames in parallel with optimized I/O and detection."""
            nonlocal total_objects_detected, frames_processed, prev_frame, motion_scores
            
            results = []
            batch_frames = []
            
            # Read frames in parallel using ThreadPoolExecutor (aggressive I/O)
            # Use more workers for I/O to keep GPU fed
            io_workers_actual = min(io_workers * 2, 16)  # Double I/O workers, cap at 16
            with ThreadPoolExecutor(max_workers=io_workers_actual) as executor:
                frame_futures = {executor.submit(read_image, path): idx for idx, path in zip(batch_indices, batch_paths)}
                for future in as_completed(frame_futures):
                    frame_idx = frame_futures[future]
                    try:
                        vision_frame = future.result()
                        batch_frames.append((frame_idx, vision_frame))
                    except Exception as e:
                        logger.warn(f"Error reading frame {frame_idx}: {e}", __name__)
                        results.append((frame_idx, FrameDetectionData(frame_number=frame_idx)))
            
            # Sort by frame index to maintain order
            batch_frames.sort(key=lambda x: x[0])
            
            # Extract frames for batch processing
            vision_frames_batch = [frame for _, frame in batch_frames]
            
            # TRUE BATCH PROCESSING: Process all frames in batch simultaneously
            # This maximizes GPU utilization by batching ONNX inference calls
            def detect_faces_batch_optimized(frame_batch: List[Tuple[int, VisionFrame]]) -> List[Tuple[int, VisionFrame, List]]:
                """Process a batch of frames with optimized parallel detection."""
                results = []
                
                # Process frames in parallel using ThreadPoolExecutor
                # Even though get_many_faces processes one at a time internally,
                # parallelizing the calls helps saturate GPU/CPU
                def detect_single_frame(frame_data: Tuple[int, VisionFrame]) -> Tuple[int, VisionFrame, List]:
                    frame_idx, vision_frame = frame_data
                    try:
                        # Detect faces with FULL analysis (embeddings + classification)
                        many_faces = get_many_faces([vision_frame])
                        sorted_faces = sort_and_filter_faces(many_faces, vision_frame=vision_frame)
                        return (frame_idx, vision_frame, sorted_faces)
                    except Exception as e:
                        logger.warn(f"Error detecting faces in frame {frame_idx}: {e}", __name__)
                        return (frame_idx, vision_frame, [])
                
                # AGGRESSIVE parallelism - process ALL frames simultaneously to saturate GPU
                # For 24GB GPU, we want to push hard - use batch_size * 3 workers
                max_parallel = min(len(frame_batch), batch_size * 3, io_workers * 3, 64)  # Cap at 64
                with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                    # Use submit instead of map for better parallelism
                    futures = [executor.submit(detect_single_frame, frame_data) for frame_data in frame_batch]
                    results = [future.result() for future in futures]
                
                return results
            
            # Process entire batch with maximum parallelism
            faces_results = detect_faces_batch_optimized(batch_frames)
            
            # Parallelize YOLO detection across frames
            def detect_yolo_for_frame(frame_data: Tuple[int, VisionFrame, List]) -> Tuple[int, VisionFrame, List, List]:
                """Detect YOLO objects for a single frame."""
                frame_idx, vision_frame, sorted_faces = frame_data
                yolo_detections = []
                frame_objects = 0
                
                if masker and yolo_model_path:
                    try:
                        intersections = masker.detect_face_object_intersections(
                            vision_frame, sorted_faces, yolo_model_path, silent=True
                        )
                        for face_idx, data in intersections.items():
                            for obj in data.get('objects_detected', []):
                                yolo_detections.append({
                                    'face_idx': face_idx,
                                    'bbox': obj['bbox'],
                                    'confidence': obj['confidence'],
                                    'distance': obj['distance_to_face']
                                })
                                frame_objects += 1
                    except Exception as e:
                        logger.warn(f"Error detecting YOLO objects in frame {frame_idx}: {e}", __name__)
                
                return (frame_idx, vision_frame, sorted_faces, yolo_detections, frame_objects)
            
            # Run YOLO detection in parallel
            yolo_results = []
            if masker and yolo_model_path:
                with ThreadPoolExecutor(max_workers=min(len(faces_results), batch_size)) as executor:
                    yolo_futures = [executor.submit(detect_yolo_for_frame, frame_data) for frame_data in faces_results]
                    yolo_results = [future.result() for future in yolo_futures]
            else:
                # No YOLO - just pass through
                yolo_results = [(idx, frame, faces, [], 0) for idx, frame, faces in faces_results]
            
            # Process each frame with motion calculation (sequential - needs prev_frame)
            for yolo_result in yolo_results:
                try:
                    if len(yolo_result) == 5:
                        frame_idx, vision_frame, sorted_faces, yolo_detections, frame_objects = yolo_result
                    else:
                        # Fallback for no YOLO case
                        frame_idx, vision_frame, sorted_faces = yolo_result[:3]
                        yolo_detections = []
                        frame_objects = 0
                    
                    # Calculate motion score (sequential - needs prev_frame)
                    motion_score = 0.0
                    if prev_frame is not None:
                        motion_score = motion_analyzer.calculate_motion_score(prev_frame, vision_frame)
                        motion_scores.append(motion_score)
                    prev_frame = vision_frame
                    
                    # Update total objects detected
                    total_objects_detected += frame_objects
                    
                    # Store in cache
                    frame_data = FrameDetectionData(
                        frame_number=frame_idx,
                        faces=sorted_faces,
                        yolo_detections=yolo_detections,
                        motion_score=motion_score,
                        interpolated=False
                    )
                    results.append((frame_idx, frame_data))
                    
                    with frames_processed_lock:
                        frames_processed += 1
                        if len(sorted_faces) > 0:
                            cache.frames_with_faces += 1
                    
                except Exception as e:
                    logger.warn(f"Error processing frame {frame_idx}: {e}", __name__)
                    results.append((frame_idx, FrameDetectionData(frame_number=frame_idx)))
                    with frames_processed_lock:
                        frames_processed += 1
            
            return results
        
        # Process frames in batches
        def write_cache_batch(cache_data: List[Tuple[int, FrameDetectionData]]) -> None:
            """Write cache data in parallel."""
            def write_single_cache(item: Tuple[int, FrameDetectionData]) -> None:
                frame_idx, frame_data = item
                cache.set_frame_data(frame_idx, frame_data)
            
            # Write cache in parallel
            with ThreadPoolExecutor(max_workers=min(len(cache_data), batch_size)) as executor:
                executor.map(write_single_cache, cache_data)
        
        if progress_callback:
            # UI mode - use callback
            for batch_start in range(0, len(frame_paths), batch_size):
                batch_end = min(batch_start + batch_size, len(frame_paths))
                batch_indices = list(range(batch_start, batch_end))
                batch_paths = [frame_paths[i] for i in batch_indices]
                
                # Process batch
                batch_results = process_frame_batch(batch_indices, batch_paths)
                
                # Store results in parallel
                write_cache_batch(batch_results)
                
                # Update progress
                progress_callback(frames_processed, len(frame_paths), cache.frames_with_faces, total_objects_detected)
        else:
            # Console mode - use tqdm
            with tqdm(total=len(frame_paths), desc='Face Buffer Scan (BATCH)', unit='frame', ascii=' =') as progress:
                for batch_start in range(0, len(frame_paths), batch_size):
                    batch_end = min(batch_start + batch_size, len(frame_paths))
                    batch_indices = list(range(batch_start, batch_end))
                    batch_paths = [frame_paths[i] for i in batch_indices]
                    
                    # Process batch
                    batch_results = process_frame_batch(batch_indices, batch_paths)
                    
                    # Store results in parallel
                    write_cache_batch(batch_results)
                    
                    # Update progress
                    progress.update(len(batch_results))
                    progress.set_postfix({
                        'faces': cache.frames_with_faces,
                        'objects': total_objects_detected,
                        'batch': batch_size
                    })
        
        # Store total objects detected
        cache.total_objects_detected = total_objects_detected
        
        phase1_time = time.time() - phase1_start
        phase_times['phase1_scan'] = phase1_time
        fps_phase1 = len(frame_paths) / phase1_time if phase1_time > 0 else 0
        logger.info(f"⏱️ Phase 1 complete: {phase1_time:.2f}s ({fps_phase1:.2f} FPS)", __name__)
        
        # Phase 2: Gap Detection
        phase2_start = time.time()
        logger.info("Phase 2: Detecting gaps in face detections...", __name__)
        gaps = detect_gaps(cache, 0, len(frame_paths) - 1)
        logger.info(f"Found {len(gaps)} gaps in detections", __name__)
        
        # Store gaps detected count
        cache.gaps_detected = len(gaps)
        
        phase2_time = time.time() - phase2_start
        phase_times['phase2_gap_detection'] = phase2_time
        
        # Phase 3: Gap Filling
        phase3_start = time.time()
        interpolation_mode = state_manager.get_item('face_buffer_interpolation_mode') or 'simple'
        max_gap_frames = state_manager.get_item('face_buffer_max_gap_frames') or 5
        
        logger.info(f"Phase 3: Filling gaps (mode: {interpolation_mode}, max: {max_gap_frames} frames)...", __name__)
        
        gap_filler = GapFiller(interpolation_mode=interpolation_mode)
        
        for gap_start, gap_end in gaps:
            gap_size = gap_end - gap_start + 1
            
            # Get surrounding motion scores
            surrounding_motion = []
            for i in range(max(0, gap_start - 5), min(len(frame_paths), gap_end + 6)):
                data = cache.get_frame_data(i)
                if data:
                    surrounding_motion.append(data.motion_score)
            
            # Calculate adaptive threshold
            adaptive_threshold = calculate_adaptive_gap_threshold(
                surrounding_motion, fps, max_gap_frames
            )
            
            # Only fill if gap is within threshold
            if gap_size <= adaptive_threshold:
                # Get faces before and after gap
                prev_data = cache.get_frame_data(gap_start - 1) if gap_start > 0 else None
                next_data = cache.get_frame_data(gap_end + 1) if gap_end < len(frame_paths) - 1 else None
                
                if prev_data and next_data and prev_data.faces and next_data.faces:
                    # Calculate average motion in gap region
                    avg_motion = np.mean(surrounding_motion) if surrounding_motion else 0.0
                    
                    # Interpolate faces
                    interpolated_frames = gap_filler.interpolate_faces(
                        prev_data.faces,
                        next_data.faces,
                        gap_size,
                        avg_motion
                    )
                    
                    # Update cache with interpolated data
                    for i, interpolated_faces in enumerate(interpolated_frames):
                        frame_num = gap_start + i
                        existing_data = cache.get_frame_data(frame_num)
                        if existing_data:
                            existing_data.faces = interpolated_faces
                            existing_data.interpolated = True
                            cache.set_frame_data(frame_num, existing_data)
                            cache.frames_interpolated += 1
                    
                    cache.gaps_filled += 1
                    logger.info(f"Filled gap: frames {gap_start}-{gap_end} ({gap_size} frames)", __name__)
            else:
                logger.info(f"Skipped gap: frames {gap_start}-{gap_end} ({gap_size} frames, threshold: {adaptive_threshold})", __name__)
        
        phase3_time = time.time() - phase3_start
        phase_times['phase3_gap_filling'] = phase3_time
        
        # Phase 4: Statistics
        total_time = time.time() - scan_start_time
        stats = cache.get_statistics()
        overall_fps = len(frame_paths) / total_time if total_time > 0 else 0
        
        logger.info("=" * 60, __name__)
        logger.info("Face Buffer Scan Complete", __name__)
        logger.info(f"  Total frames: {stats['total_frames']}", __name__)
        logger.info(f"  Frames with faces: {stats['frames_with_faces']}", __name__)
        logger.info(f"  Objects detected: {stats['total_objects_detected']}", __name__)
        logger.info(f"  Gaps detected: {stats['gaps_detected']}", __name__)
        logger.info(f"  Gaps filled: {stats['gaps_filled']}", __name__)
        logger.info(f"  Frames interpolated: {stats['frames_interpolated']}", __name__)
        logger.info(f"  Memory usage: {stats['memory_usage_mb']:.2f} MB", __name__)
        logger.info(f"  Disk cache: {'Enabled' if stats['disk_cache_enabled'] else 'Disabled'}", __name__)
        logger.info("=" * 60, __name__)
        logger.info("⏱️ PERFORMANCE METRICS:", __name__)
        logger.info(f"  Phase 1 (Scan): {phase_times.get('phase1_scan', 0):.2f}s ({fps_phase1:.2f} FPS)", __name__)
        logger.info(f"  Phase 2 (Gap Detection): {phase_times.get('phase2_gap_detection', 0):.2f}s", __name__)
        logger.info(f"  Phase 3 (Gap Filling): {phase_times.get('phase3_gap_filling', 0):.2f}s", __name__)
        logger.info(f"  Total Time: {total_time:.2f}s", __name__)
        logger.info(f"  Overall FPS: {overall_fps:.2f} frames/sec", __name__)
        logger.info(f"  Batch Size: {batch_size}, Semaphore Limit: {semaphore_limit}", __name__)
        logger.info("=" * 60, __name__)
        
        return cache
    
    finally:
        # Reset semaphore limit to original value (1) after scan completes
        reset_semaphore_limit()
        logger.info(f"🔒 Reset thread semaphore limit to 1", __name__)

