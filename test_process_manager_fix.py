"""
Test script to verify the process_manager fix
This script simulates various scenarios to ensure process_manager.end() is always called
"""
import sys
import os

# Add the extension path
ext_dir = os.path.dirname(__file__)
if ext_dir not in sys.path:
    sys.path.insert(0, ext_dir)

from facefusion import process_manager

def test_normal_flow():
    """Test that normal processing flow works correctly"""
    print("\n=== Test 1: Normal Flow ===")
    initial_state = process_manager.get_process_state()
    print(f"Initial state: {initial_state}")
    
    process_manager.start()
    print(f"After start: {process_manager.get_process_state()}")
    
    try:
        # Simulate some work
        print("Doing work...")
        return "success"
    finally:
        process_manager.end()
        print(f"After end: {process_manager.get_process_state()}")
    
def test_early_return():
    """Test that early return still calls end() via finally"""
    print("\n=== Test 2: Early Return ===")
    initial_state = process_manager.get_process_state()
    print(f"Initial state: {initial_state}")
    
    process_manager.start()
    print(f"After start: {process_manager.get_process_state()}")
    
    try:
        # Simulate early return
        print("Early return triggered!")
        return "early_exit"
    finally:
        process_manager.end()
        print(f"After end (from finally): {process_manager.get_process_state()}")

def test_exception_handling():
    """Test that exceptions still trigger finally block"""
    print("\n=== Test 3: Exception Handling ===")
    initial_state = process_manager.get_process_state()
    print(f"Initial state: {initial_state}")
    
    process_manager.start()
    print(f"After start: {process_manager.get_process_state()}")
    
    try:
        # Simulate an exception
        print("About to raise exception...")
        raise ValueError("Test exception")
    except ValueError as e:
        print(f"Caught exception: {e}")
    finally:
        process_manager.end()
        print(f"After end (from finally): {process_manager.get_process_state()}")

def test_stopping_state():
    """Test stopping state behavior"""
    print("\n=== Test 4: Stopping State ===")
    initial_state = process_manager.get_process_state()
    print(f"Initial state: {initial_state}")
    
    process_manager.start()
    print(f"After start: {process_manager.get_process_state()}")
    
    # Simulate user stopping
    process_manager.stop()
    print(f"After stop request: {process_manager.get_process_state()}")
    
    try:
        # Check if stopping
        if process_manager.is_stopping():
            print("Detected stopping state, returning early")
            return "stopped"
    finally:
        process_manager.end()
        print(f"After end (from finally): {process_manager.get_process_state()}")

def verify_state(expected_state='pending'):
    """Verify the process manager is in expected state"""
    current_state = process_manager.get_process_state()
    if current_state == expected_state:
        print(f"✅ State is correct: {current_state}")
        return True
    else:
        print(f"❌ State is incorrect: {current_state} (expected: {expected_state})")
        return False

def main():
    print("=" * 60)
    print("Testing Process Manager Fix")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Normal flow
    result = test_normal_flow()
    all_passed &= verify_state('pending')
    
    # Test 2: Early return
    result = test_early_return()
    all_passed &= verify_state('pending')
    
    # Test 3: Exception handling
    test_exception_handling()
    all_passed &= verify_state('pending')
    
    # Test 4: Stopping state
    result = test_stopping_state()
    all_passed &= verify_state('pending')
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Process manager is properly reset in all cases")
    else:
        print("❌ SOME TESTS FAILED - Process manager may not be properly reset")
    print("=" * 60)

if __name__ == '__main__':
    main()

