import threading
from contextlib import nullcontext
from typing import ContextManager, Union, Optional

from facefusion.execution import has_execution_provider

THREAD_LOCK: threading.Lock = threading.Lock()
THREAD_SEMAPHORE: threading.Semaphore = threading.Semaphore()
NULL_CONTEXT: ContextManager[None] = nullcontext()
_ORIGINAL_SEMAPHORE_LIMIT: int = 1


def thread_lock() -> threading.Lock:
    return THREAD_LOCK


def thread_semaphore() -> threading.Semaphore:
    return THREAD_SEMAPHORE


def conditional_thread_semaphore() -> Union[threading.Semaphore, ContextManager[None]]:
    if has_execution_provider('directml') or has_execution_provider('rocm'):
        return THREAD_SEMAPHORE
    return NULL_CONTEXT


def set_semaphore_limit(limit: int) -> None:
    """
    Set the thread semaphore limit for parallel processing.
    Higher limits allow more concurrent ONNX inference calls.
    
    Args:
        limit: Maximum number of concurrent threads allowed (default: 1)
    """
    global THREAD_SEMAPHORE, _ORIGINAL_SEMAPHORE_LIMIT
    
    # Store original limit if not already stored
    if _ORIGINAL_SEMAPHORE_LIMIT == 1:
        _ORIGINAL_SEMAPHORE_LIMIT = THREAD_SEMAPHORE._value if hasattr(THREAD_SEMAPHORE, '_value') else 1
    
    # Create new semaphore with desired limit
    THREAD_SEMAPHORE = threading.Semaphore(limit)


def reset_semaphore_limit() -> None:
    """Reset thread semaphore to original limit (typically 1)."""
    global THREAD_SEMAPHORE, _ORIGINAL_SEMAPHORE_LIMIT
    THREAD_SEMAPHORE = threading.Semaphore(_ORIGINAL_SEMAPHORE_LIMIT)
