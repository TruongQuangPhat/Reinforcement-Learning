"""Profiling helpers for experiment runtime and memory usage."""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ProfileResult(Generic[T]):
    """Result returned by `profile_callable`."""

    result: T
    wall_time_sec: float
    cpu_time_sec: float
    current_memory_mb: float
    peak_memory_mb: float


def profile_callable(func: Callable[..., T], *args: Any, **kwargs: Any) -> ProfileResult[T]:
    """Run a callable and collect wall-clock, CPU, and tracemalloc metrics."""
    tracemalloc.start()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        result = func(*args, **kwargs)
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        wall_end = time.perf_counter()
        cpu_end = time.process_time()
        tracemalloc.stop()

    return ProfileResult(
        result=result,
        wall_time_sec=wall_end - wall_start,
        cpu_time_sec=cpu_end - cpu_start,
        current_memory_mb=current_bytes / (1024 * 1024),
        peak_memory_mb=peak_bytes / (1024 * 1024),
    )
