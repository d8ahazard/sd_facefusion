"""
Quick A/B helper: run two short Face Swapper passes on extracted temp frames.

Usage (from stable-diffusion-webui root, venv active):
  python extensions/sd_facefusion/scripts/benchmark_swap_threads.py ^
    --frames-dir "outputs/facefusion/temp/YOUR_VIDEO" ^
    --count 400 ^
    --threads 8

  python extensions/sd_facefusion/scripts/benchmark_swap_threads.py ^
    --frames-dir "outputs/facefusion/temp/YOUR_VIDEO" ^
    --count 400 ^
    --threads 22

Compare the reported frame/s lines. Requires frames already extracted (keep_temp or partial run).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

EXTENSION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if EXTENSION_ROOT not in sys.path:
    sys.path.insert(0, EXTENSION_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark face swap throughput by thread count')
    parser.add_argument('--frames-dir', required=True, help='Directory of %08d.png temp frames')
    parser.add_argument('--count', type=int, default=400, help='Number of frames to process')
    parser.add_argument('--threads', type=int, required=True, help='execution_thread_count for this run')
    args = parser.parse_args()

    frames = sorted(
        f for f in os.listdir(args.frames_dir)
        if f.lower().endswith('.png')
    )[: args.count]
    if not frames:
        raise SystemExit(f'No PNG frames in {args.frames_dir}')

    from facefusion import state_manager
    from facefusion.processors.core import get_processors_modules

    state_manager.init_item('processors', ['face_swapper'])
    state_manager.init_item('execution_thread_count', args.threads)
    state_manager.init_item('execution_queue_count', 2)
    state_manager.init_item('execution_providers', ['cuda', 'cpu'])
    state_manager.init_item('face_selector_mode', 'reference')
    state_manager.init_item('log_level', 'error')

    paths = [os.path.join(args.frames_dir, name) for name in frames]
    swapper = get_processors_modules(['face_swapper'])[0]
    if not swapper.pre_process('output'):
        raise SystemExit('pre_process failed — set target_path/sources in state or run from UI first')

    started = time.perf_counter()
    swapper.process_video(paths)
    elapsed = time.perf_counter() - started
    fps = len(paths) / elapsed if elapsed > 0 else 0.0
    print(f'threads={args.threads} frames={len(paths)} elapsed={elapsed:.1f}s fps={fps:.2f}')


if __name__ == '__main__':
    main()
