"""RAM-aware parallelism helpers for the per-target metrics loop.

Targets are independent, so the per-target work is parallelised with a thread
pool: threads share the (read-only) loaded corpus, event index and precomputes,
so peak RAM stays roughly constant regardless of the worker count — unlike a
process pool, where each worker would copy the multi-GB base (measured ~12 GB)
and exhaust memory. ``resolve_n_jobs`` keeps the worker count within available
RAM and falls back to serial when memory is tight or unknown.
"""

from __future__ import annotations

import contextlib
import os

# Transient per-target working set on top of the shared base. Measured on the real
# GRG corpus (2026-06): shared base ~12 GB; per-target peak is dominated by the
# Citation-Identity pair materialization at ~4 GB. resolve_n_jobs("auto") sizes the
# worker count against this, so a too-low estimate here causes OOM under parallelism.
DEFAULT_PER_WORKER_BYTES = 4_500_000_000


def _free_ram_bytes() -> int | None:
    """Best-effort available physical RAM in bytes; None if it cannot be determined."""
    try:  # Windows
        import ctypes

        if hasattr(ctypes, "windll"):

            class _MemStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemStatusEx()
            stat.dwLength = ctypes.sizeof(_MemStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
    except Exception:
        pass
    try:  # POSIX (Linux/macOS)
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        pass
    return None


def resolve_n_jobs(n_jobs, per_worker_bytes: int = DEFAULT_PER_WORKER_BYTES) -> int:
    """Resolve ``n_jobs`` (int or "auto") to a concrete, RAM-safe worker count.

    "auto" = min(cpu_count, free_RAM*0.6 / per_worker_bytes); unknown RAM -> 1.
    A negative int counts back from cpu_count (-1 = all cores). Always >= 1.
    """
    cores = os.cpu_count() or 1
    if isinstance(n_jobs, bool):  # guard: bool is an int subclass
        raise TypeError("n_jobs must be an int or 'auto', not bool")
    if isinstance(n_jobs, int):
        if n_jobs < 0:
            return max(1, cores + 1 + n_jobs)
        return max(1, n_jobs)
    if n_jobs != "auto":
        raise ValueError(f"n_jobs must be an int or 'auto', got {n_jobs!r}")
    free = _free_ram_bytes()
    if not free:
        return 1
    ram_cap = int(free * 0.6 // per_worker_bytes)
    return max(1, min(cores, ram_cap))


@contextlib.contextmanager
def limit_blas_threads(limit: int = 1):
    """Cap intra-op BLAS threads during parallel sections to avoid oversubscription.

    Uses threadpoolctl if available (it ships with scikit-learn); otherwise a no-op.
    """
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        yield
        return
    with threadpool_limits(limits=limit):
        yield
