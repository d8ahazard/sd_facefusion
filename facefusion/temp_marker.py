"""
Temp frame marker system - tracks extraction state and modifications.
Prevents unnecessary re-extraction when frames are already available.
"""

import json
import os
import hashlib
from pathlib import Path
from typing import Dict, Optional

from facefusion.temp_helper import get_temp_directory_path


def get_marker_path(target_path: str) -> Path:
    """Get the marker file path for a target video."""
    temp_dir = get_temp_directory_path(target_path)
    return Path(temp_dir) / '.frame_marker.json'


def create_extraction_marker(
    target_path: str, 
    resolution: str, 
    fps: float,
    frame_count: int
) -> None:
    """
    Create a marker file indicating frames have been extracted.
    
    Args:
        target_path: Path to the video file
        resolution: Extraction resolution (e.g., '1920x1080')
        fps: Extraction FPS
        frame_count: Number of frames extracted
    """
    marker_path = get_marker_path(target_path)
    
    # Calculate video file hash for validation
    video_hash = _get_file_hash(target_path)
    
    marker_data = {
        'video_path': target_path,
        'video_hash': video_hash,
        'resolution': resolution,
        'fps': fps,
        'frame_count': frame_count,
        'extracted': True,
        'processed': False,
        'timestamp': os.path.getmtime(target_path)
    }
    
    try:
        with open(marker_path, 'w') as f:
            json.dump(marker_data, f, indent=2)
    except Exception as e:
        # Marker is optional, don't fail if we can't create it
        pass


def mark_frames_processed(target_path: str) -> None:
    """Mark that frames have been processed/modified."""
    marker_path = get_marker_path(target_path)
    
    if not marker_path.exists():
        return
    
    try:
        with open(marker_path, 'r') as f:
            marker_data = json.load(f)
        
        marker_data['processed'] = True
        
        with open(marker_path, 'w') as f:
            json.dump(marker_data, f, indent=2)
    except Exception as e:
        pass


def are_frames_valid(
    target_path: str, 
    required_resolution: str, 
    required_fps: float
) -> bool:
    """
    Check if extracted frames are valid and match requirements.
    
    Returns:
        True if frames exist and match requirements, False otherwise
    """
    marker_path = get_marker_path(target_path)
    
    if not marker_path.exists():
        return False
    
    try:
        with open(marker_path, 'r') as f:
            marker_data = json.load(f)
        
        # Verify video hasn't changed
        current_hash = _get_file_hash(target_path)
        if marker_data.get('video_hash') != current_hash:
            return False
        
        # Verify timestamp
        current_timestamp = os.path.getmtime(target_path)
        if marker_data.get('timestamp') != current_timestamp:
            return False
        
        # Verify extraction parameters match
        if marker_data.get('resolution') != required_resolution:
            return False
        
        if marker_data.get('fps') != required_fps:
            return False
        
        # Verify frames haven't been processed/modified yet
        if marker_data.get('processed', False):
            return False
        
        return marker_data.get('extracted', False)
        
    except Exception as e:
        return False


def _get_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """Calculate MD5 hash of file (first 1MB only for speed)."""
    try:
        hasher = hashlib.md5()
        bytes_read = 0
        max_bytes = 1024 * 1024  # 1MB
        
        with open(file_path, 'rb') as f:
            while bytes_read < max_bytes:
                chunk = f.read(min(chunk_size, max_bytes - bytes_read))
                if not chunk:
                    break
                hasher.update(chunk)
                bytes_read += len(chunk)
        
        return hasher.hexdigest()
    except Exception as e:
        return ""


def get_marker_info(target_path: str) -> Optional[Dict]:
    """Get marker information if it exists."""
    marker_path = get_marker_path(target_path)
    
    if not marker_path.exists():
        return None
    
    try:
        with open(marker_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None

