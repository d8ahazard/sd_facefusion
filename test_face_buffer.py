"""
Test script for Face Buffer System

This script demonstrates and validates the face buffer functionality.
Run this to test the face buffer with sample data.
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from facefusion import state_manager
from facefusion.face_buffer import (
    FaceBufferCache,
    FrameDetectionData,
    GapFiller,
    MotionAnalyzer,
    detect_gaps,
    calculate_adaptive_gap_threshold
)
from facefusion.face_buffer_config import (
    initialize_face_buffer_config,
    get_face_buffer_config,
    is_face_buffer_enabled,
    set_face_buffer_enabled
)


def test_face_buffer_cache():
    """Test FaceBufferCache creation and basic operations."""
    print("=" * 60)
    print("Test 1: FaceBufferCache Creation and Operations")
    print("=" * 60)
    
    # Create a cache
    cache = FaceBufferCache(
        video_path="test_video.mp4",
        memory_limit_mb=512,
        disk_cache_enabled=False,  # Disable disk for unit test
        max_memory_frames=10
    )
    
    # Add some test data
    for i in range(15):
        frame_data = FrameDetectionData(
            frame_number=i,
            faces=[],  # Empty faces for this test
            motion_score=0.1 * i
        )
        cache.set_frame_data(i, frame_data)
    
    # Verify memory limit (should evict to disk if enabled)
    print(f"✓ Added 15 frames to cache")
    print(f"  Memory cache size: {len(cache.memory_cache)}")
    print(f"  Memory usage: {cache.memory_usage_mb:.2f} MB")
    
    # Retrieve data
    data = cache.get_frame_data(5)
    assert data is not None, "Failed to retrieve frame data"
    assert data.frame_number == 5, "Retrieved wrong frame"
    print(f"✓ Successfully retrieved frame {data.frame_number}")
    
    # Get statistics
    stats = cache.get_statistics()
    print(f"✓ Cache statistics:")
    for key, value in stats.items():
        print(f"    {key}: {value}")
    
    print("✓ Test 1 PASSED\n")


def test_motion_analyzer():
    """Test MotionAnalyzer with synthetic frames."""
    print("=" * 60)
    print("Test 2: Motion Analysis")
    print("=" * 60)
    
    import numpy as np
    
    analyzer = MotionAnalyzer()
    
    # Create two similar frames (low motion)
    frame1 = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
    frame2 = frame1 + np.random.randint(-5, 5, (480, 640, 3), dtype=np.uint8)
    
    motion_score = analyzer.calculate_motion_score(frame1, frame2)
    print(f"✓ Low motion score: {motion_score:.4f}")
    assert motion_score < 0.3, "Low motion should have low score"
    
    # Create two very different frames (high motion)
    frame3 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame4 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    motion_score2 = analyzer.calculate_motion_score(frame3, frame4)
    print(f"✓ High motion score: {motion_score2:.4f}")
    assert motion_score2 > motion_score, "High motion should have higher score"
    
    print("✓ Test 2 PASSED\n")


def test_gap_detection():
    """Test gap detection logic."""
    print("=" * 60)
    print("Test 3: Gap Detection")
    print("=" * 60)
    
    from facefusion.typing import Face, FaceLandmarkSet, FaceScoreSet
    
    # Create a cache with gaps
    cache = FaceBufferCache(
        video_path="test_video.mp4",
        disk_cache_enabled=False
    )
    
    # Add frames with gaps (no faces in frames 3-5 and 8-9)
    for i in range(12):
        has_faces = i not in [3, 4, 5, 8, 9]
        faces = []
        
        if has_faces:
            # Create a dummy face
            faces = [Face(
                bounding_box=(100, 100, 200, 200),
                score_set=FaceScoreSet(detector=0.9, landmarker=0.8),
                landmark_set=FaceLandmarkSet(),
                angle=0.0,
                embedding=None,
                normed_embedding=None,
                gender='male',
                age=range(25, 35),
                race='white'
            )]
        
        frame_data = FrameDetectionData(
            frame_number=i,
            faces=faces
        )
        cache.set_frame_data(i, frame_data)
    
    # Detect gaps
    gaps = detect_gaps(cache, 0, 11)
    print(f"✓ Detected {len(gaps)} gaps: {gaps}")
    
    assert len(gaps) == 2, f"Expected 2 gaps, found {len(gaps)}"
    assert gaps[0] == (3, 5), f"First gap should be (3, 5), got {gaps[0]}"
    assert gaps[1] == (8, 9), f"Second gap should be (8, 9), got {gaps[1]}"
    
    print("✓ Test 3 PASSED\n")


def test_adaptive_threshold():
    """Test adaptive gap threshold calculation."""
    print("=" * 60)
    print("Test 4: Adaptive Gap Threshold")
    print("=" * 60)
    
    # Low motion, high FPS → larger threshold
    low_motion_scores = [0.05, 0.06, 0.04, 0.05]
    threshold1 = calculate_adaptive_gap_threshold(low_motion_scores, 60.0, 10)
    print(f"✓ Low motion (60fps, max=10): threshold = {threshold1}")
    
    # High motion, low FPS → smaller threshold
    high_motion_scores = [0.8, 0.9, 0.85, 0.88]
    threshold2 = calculate_adaptive_gap_threshold(high_motion_scores, 24.0, 10)
    print(f"✓ High motion (24fps, max=10): threshold = {threshold2}")
    
    assert threshold1 > threshold2, "Low motion should allow larger gaps"
    print(f"✓ Adaptive threshold working correctly: {threshold1} > {threshold2}")
    
    print("✓ Test 4 PASSED\n")


def test_configuration():
    """Test face buffer configuration."""
    print("=" * 60)
    print("Test 5: Configuration Management")
    print("=" * 60)
    
    # Initialize config
    initialize_face_buffer_config()
    
    # Get config
    config = get_face_buffer_config()
    print(f"✓ Default configuration:")
    for key, value in config.items():
        print(f"    {key}: {value}")
    
    # Test enabled flag
    initial_state = is_face_buffer_enabled()
    print(f"✓ Initial enabled state: {initial_state}")
    
    set_face_buffer_enabled(True)
    assert is_face_buffer_enabled() == True, "Failed to enable face buffer"
    print(f"✓ Successfully enabled face buffer")
    
    set_face_buffer_enabled(False)
    assert is_face_buffer_enabled() == False, "Failed to disable face buffer"
    print(f"✓ Successfully disabled face buffer")
    
    # Restore initial state
    set_face_buffer_enabled(initial_state)
    
    print("✓ Test 5 PASSED\n")


def test_face_interpolation():
    """Test face interpolation logic."""
    print("=" * 60)
    print("Test 6: Face Interpolation")
    print("=" * 60)
    
    import numpy as np
    from facefusion.typing import Face, FaceLandmarkSet, FaceScoreSet
    
    gap_filler = GapFiller(interpolation_mode='simple')
    
    # Create two test faces
    landmark_5 = np.array([[100, 120], [140, 120], [120, 140], [110, 160], [130, 160]])
    
    face1 = Face(
        bounding_box=(100, 100, 200, 200),
        score_set=FaceScoreSet(detector=0.9, landmarker=0.8),
        landmark_set={'5': landmark_5.copy()},
        angle=0.0,
        embedding=np.random.randn(512),
        normed_embedding=None,
        gender='male',
        age=range(25, 35),
        race='white'
    )
    face1.normed_embedding = face1.embedding / np.linalg.norm(face1.embedding)
    
    face2 = Face(
        bounding_box=(120, 110, 220, 210),
        score_set=FaceScoreSet(detector=0.9, landmarker=0.8),
        landmark_set={'5': landmark_5 + 20},
        angle=5.0,
        embedding=np.random.randn(512),
        normed_embedding=None,
        gender='male',
        age=range(25, 35),
        race='white'
    )
    face2.normed_embedding = face2.embedding / np.linalg.norm(face2.embedding)
    
    # Interpolate across 3 frames
    interpolated = gap_filler.interpolate_faces([face1], [face2], gap_size=3, motion_score=0.2)
    
    print(f"✓ Interpolated {len(interpolated)} frames")
    assert len(interpolated) == 3, f"Expected 3 interpolated frames, got {len(interpolated)}"
    
    for i, faces in enumerate(interpolated):
        print(f"  Frame {i+1}: {len(faces)} face(s)")
        if faces:
            bbox = faces[0].bounding_box
            print(f"    Bounding box: ({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})")
    
    print("✓ Test 6 PASSED\n")


def run_all_tests():
    """Run all face buffer tests."""
    print("\n" + "=" * 60)
    print("FACE BUFFER SYSTEM - TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        test_face_buffer_cache,
        test_motion_analyzer,
        test_gap_detection,
        test_adaptive_threshold,
        test_configuration,
        test_face_interpolation
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

