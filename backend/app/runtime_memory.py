"""Return unused allocator arenas after bounded jobs on small Linux containers."""

import gc
import os
import sys


def release_idle_memory():
    if (
        sys.platform != "linux"
        or os.getenv("SEMATRACE_RELEASE_IDLE_MEMORY", "false") != "true"
    ):
        return
    gc.collect()
    try:
        import ctypes

        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trim(0)
    except (OSError, AttributeError):
        # Non-glibc runtimes retain their normal allocator behavior.
        pass
