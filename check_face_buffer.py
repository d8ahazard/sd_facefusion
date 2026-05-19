"""
Quick script to check if face buffer is enabled and working.
Run this to verify your configuration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facefusion import state_manager
from facefusion.face_buffer_config import (
    is_face_buffer_enabled,
    get_face_buffer_config,
    set_face_buffer_enabled
)

print("=" * 60)
print("FACE BUFFER STATUS CHECK")
print("=" * 60)

# Check current status
enabled = is_face_buffer_enabled()
print(f"\n✓ Face Buffer Enabled: {enabled}")

if not enabled:
    print("\n⚠ Face buffer is currently DISABLED")
    print("\nTo enable it, add this to your code:")
    print("  from facefusion import state_manager")
    print("  state_manager.set_item('face_buffer_enabled', True)")
    print("\nOr run:")
    print("  python check_face_buffer.py --enable")
    
    if '--enable' in sys.argv:
        print("\n>>> Enabling face buffer now...")
        set_face_buffer_enabled(True)
        print("✓ Face buffer ENABLED")
        enabled = True

if enabled:
    print("\n✓ Face buffer is ENABLED")
    print("\nCurrent Configuration:")
    config = get_face_buffer_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("WHAT TO EXPECT WHEN PROCESSING VIDEO:")
    print("=" * 60)
    print("1. After frame extraction, you'll see:")
    print("   'Face buffer: Pre-scanning X frames...'")
    print("\n2. Progress updates every 100 frames:")
    print("   'Scanned 100/1234 frames'")
    print("\n3. Gap detection summary:")
    print("   'Found X gaps in detections'")
    print("\n4. Final statistics:")
    print("   'Face Buffer Scan Complete'")
    print("   '  Frames interpolated: X'")
    print("   '  Gaps filled: X'")
    
    print("\n" + "=" * 60)
    print("VERIFICATION TIPS:")
    print("=" * 60)
    print("- Check console output during video processing")
    print("- Compare processing time with/without face buffer")
    print("- Look for 'cached_faces' in debug logs")
    print("- Video output should have less flickering")

print("\n" + "=" * 60)
print("MODULES CHECK:")
print("=" * 60)

try:
    from facefusion.face_buffer import FaceBufferCache
    print("✓ face_buffer.py imported successfully")
    print(f"  FaceBufferCache class available")
except ImportError as e:
    print(f"✗ Failed to import face_buffer: {e}")

try:
    from facefusion.face_buffer_config import initialize_face_buffer_config
    print("✓ face_buffer_config.py imported successfully")
except ImportError as e:
    print(f"✗ Failed to import face_buffer_config: {e}")

print("\n" + "=" * 60)
print("STATUS: " + ("✓ READY TO USE" if enabled else "⚠ NEEDS ENABLING"))
print("=" * 60)

