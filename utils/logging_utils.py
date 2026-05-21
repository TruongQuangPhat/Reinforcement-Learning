"""Small print-based helpers for experiment progress output."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def log_message(message: str, verbose: int, min_verbose: int = 1) -> None:
    """Print a message when verbosity is high enough."""
    if verbose >= min_verbose:
        print(message, flush=True)


def print_algorithm_start(group: str, algorithm_name: str, verbose: int) -> None:
    """Print a compact algorithm start line."""
    log_message(f"[{group}] Running {algorithm_name}...", verbose)


def print_metric_block(title: str, metrics: Sequence[tuple[str, Any]], verbose: int) -> None:
    """Print an aligned metric block."""
    if verbose < 1:
        return
    print(title, flush=True)
    width = max((len(name) for name, _ in metrics), default=0)
    for name, value in metrics:
        print(f"  {name:<{width}} : {value}", flush=True)


def print_progress_header(
    title: str,
    columns: Sequence[str],
    widths: Sequence[int],
    verbose: int,
) -> None:
    """Print a table-style progress header."""
    if verbose < 2:
        return
    print(title, flush=True)
    print_progress_row(columns, widths, verbose, align="left")
    separators = ["-" * width for width in widths]
    print_progress_row(separators, widths, verbose, align="left")


def print_progress_row(
    values: Sequence[Any],
    widths: Sequence[int],
    verbose: int,
    align: str = "right",
) -> None:
    """Print one aligned progress row."""
    if verbose < 2:
        return
    cells: list[str] = []
    for value, width in zip(values, widths):
        text = str(value)
        cells.append(f"{text:<{width}}" if align == "left" else f"{text:>{width}}")
    print("  " + " | ".join(cells), flush=True)


def print_table(
    title: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    widths: Sequence[int],
    verbose: int,
) -> None:
    """Print a compact table when verbosity is enabled."""
    if verbose < 1:
        return
    print(title, flush=True)
    _print_table_row(columns, widths, align="left")
    _print_table_row(["-" * width for width in widths], widths, align="left")
    for row in rows:
        _print_table_row(row, widths)


def _print_table_row(values: Sequence[Any], widths: Sequence[int], align: str = "right") -> None:
    """Print one table row regardless of verbosity."""
    cells: list[str] = []
    for value, width in zip(values, widths):
        text = str(value)
        cells.append(f"{text:<{width}}" if align == "left" else f"{text:>{width}}")
    print("  " + " | ".join(cells), flush=True)


def format_float(value: Any, precision: int = 3) -> str:
    """Format numeric values for human-readable output."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def format_scientific(value: Any, precision: int = 3) -> str:
    """Format numeric values in scientific notation."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{precision}e}"
    except (TypeError, ValueError):
        return str(value)
