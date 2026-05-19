"""
Configuration helpers for Face Buffer system.
"""

from typing import Any, Dict

from facefusion import state_manager


# Default configuration values
DEFAULT_CONFIG = {
    'face_buffer_enabled': False,
    'face_buffer_interpolation_mode': 'simple',  # 'simple', 'optical_flow', 'auto'
    'face_buffer_max_gap_frames': 5,
    'face_buffer_memory_limit_mb': 1024,
    'face_buffer_disk_cache': True,
    'face_buffer_batch_size': None,  # Auto-tuned on first boot
    'face_buffer_io_workers': None,  # Auto-tuned on first boot
    'face_buffer_benchmark_done': False
}


def initialize_face_buffer_config() -> None:
    """Initialize face buffer configuration with defaults if not already set."""
    for key, default_value in DEFAULT_CONFIG.items():
        if state_manager.get_item(key) is None:
            state_manager.set_item(key, default_value)


def get_face_buffer_config() -> Dict[str, Any]:
    """Get current face buffer configuration."""
    return {
        key: state_manager.get_item(key) or default_value
        for key, default_value in DEFAULT_CONFIG.items()
    }


def is_face_buffer_enabled() -> bool:
    """Check if face buffer is enabled."""
    enabled = state_manager.get_item('face_buffer_enabled')
    return enabled if enabled is not None else False


def set_face_buffer_enabled(enabled: bool) -> None:
    """Enable or disable face buffer."""
    state_manager.set_item('face_buffer_enabled', enabled)


def get_interpolation_mode() -> str:
    """Get current interpolation mode."""
    mode = state_manager.get_item('face_buffer_interpolation_mode')
    return mode if mode else 'simple'


def set_interpolation_mode(mode: str) -> None:
    """Set interpolation mode."""
    valid_modes = ['simple', 'optical_flow', 'auto']
    if mode not in valid_modes:
        raise ValueError(f"Invalid interpolation mode: {mode}. Must be one of {valid_modes}")
    state_manager.set_item('face_buffer_interpolation_mode', mode)


def get_batch_size() -> int:
    """Get optimal batch size (auto-tuned or default)."""
    batch_size = state_manager.get_item('face_buffer_batch_size')
    if batch_size is None:
        # Aggressive defaults for high-end GPUs
        import torch
        if torch.cuda.is_available():
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                # 24GB GPU = batch 32+, 16GB = batch 24, 8GB = batch 16, 4GB = batch 8
                if gpu_memory_gb >= 20:
                    return 32  # Aggressive for 24GB+ GPUs
                elif gpu_memory_gb >= 12:
                    return 24
                elif gpu_memory_gb >= 6:
                    return 16
                else:
                    return 8
            except:
                return 16  # Safe default for GPU
        else:
            # CPU only - smaller batches
            return 4
    return batch_size


def set_batch_size(batch_size: int) -> None:
    """Set batch size."""
    state_manager.set_item('face_buffer_batch_size', batch_size)


def get_io_workers() -> int:
    """Get optimal I/O worker count (auto-tuned or default)."""
    workers = state_manager.get_item('face_buffer_io_workers')
    if workers is None:
        # Aggressive defaults - more workers for better I/O throughput
        import os
        cpu_count = os.cpu_count() or 4
        # Use more workers: 4-12 depending on CPU cores
        return min(12, max(4, cpu_count // 2))
    return workers


def reset_benchmark() -> None:
    """Reset benchmark cache to force re-tuning on next scan."""
    state_manager.set_item('face_buffer_benchmark_done', False)
    state_manager.set_item('face_buffer_batch_size', None)
    state_manager.set_item('face_buffer_io_workers', None)


def run_benchmark() -> Dict[str, int]:
    """
    Quick benchmark to determine optimal batch size and I/O workers.
    Runs on first boot and caches results.
    Returns dict with 'batch_size' and 'io_workers'.
    """
    from facefusion import logger
    import time
    import numpy as np
    import torch
    from concurrent.futures import ThreadPoolExecutor
    
    logger.info("Running face buffer performance benchmark (quick test)...", __name__)
    
    # Create dummy frames for testing (smaller for quick benchmark)
    test_frames = []
    for i in range(10):  # Test with 10 frames for speed
        # Create a simple test frame (640x480 RGB)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_frames.append(frame)
    
    # Test different batch sizes (aggressive for high-end GPUs)
    batch_sizes = [4, 8, 16]
    if torch.cuda.is_available():
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb >= 20:
                batch_sizes = [16, 24, 32, 48]  # Aggressive for 24GB+ GPUs
            elif gpu_memory_gb >= 12:
                batch_sizes = [8, 16, 24, 32]
            else:
                batch_sizes = [4, 8, 16, 24]
        except:
            batch_sizes = [8, 16, 24, 32]
    
    best_batch_size = 4
    best_throughput = 0.0
    
    logger.info("Testing batch processing performance...", __name__)
    for batch_size in batch_sizes:
        try:
            start_time = time.time()
            # Simulate batch processing with parallel I/O
            def process_batch_sim(batch):
                # Simulate I/O + processing overhead
                time.sleep(0.005 * len(batch))  # Rough simulation
                return len(batch)
            
            with ThreadPoolExecutor(max_workers=min(batch_size, 4)) as executor:
                futures = []
                for i in range(0, len(test_frames), batch_size):
                    batch = test_frames[i:i+batch_size]
                    futures.append(executor.submit(process_batch_sim, batch))
                for future in futures:
                    future.result()
            
            elapsed = time.time() - start_time
            throughput = len(test_frames) / elapsed if elapsed > 0 else 0
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = batch_size
            
            logger.info(f"  Batch size {batch_size}: {throughput:.1f} frames/sec (simulated)", __name__)
        except Exception as e:
            logger.warn(f"  Batch size {batch_size} failed: {e}", __name__)
            continue
    
    # Determine I/O workers based on CPU cores (more aggressive)
    import os
    cpu_count = os.cpu_count() or 4
    io_workers = min(12, max(4, cpu_count // 2))  # More workers for better throughput
    
    # Cap batch size based on GPU memory if available (aggressive)
    if torch.cuda.is_available():
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb >= 20:
                # 24GB+ GPU - allow very large batches
                best_batch_size = max(best_batch_size, 24)  # Ensure at least 24
                best_batch_size = min(best_batch_size, 48)  # Cap at 48
            elif gpu_memory_gb >= 12:
                best_batch_size = max(best_batch_size, 16)
                best_batch_size = min(best_batch_size, 32)
            elif gpu_memory_gb >= 6:
                best_batch_size = max(best_batch_size, 8)
                best_batch_size = min(best_batch_size, 24)
            else:
                best_batch_size = min(best_batch_size, 16)
            logger.info(f"GPU detected: {gpu_memory_gb:.1f}GB - using batch size {best_batch_size}", __name__)
        except:
            pass
    
    # Aggressive defaults if benchmark fails
    if best_batch_size < 4:
        best_batch_size = 16 if torch.cuda.is_available() else 4
    
    logger.info(f"✅ Benchmark complete: batch_size={best_batch_size}, io_workers={io_workers}", __name__)
    
    # Cache results
    state_manager.set_item('face_buffer_batch_size', best_batch_size)
    state_manager.set_item('face_buffer_io_workers', io_workers)
    state_manager.set_item('face_buffer_benchmark_done', True)
    
    return {
        'batch_size': best_batch_size,
        'io_workers': io_workers
    }

