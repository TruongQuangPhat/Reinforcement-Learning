"""JSON IO helpers for experiment logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.logging_utils import log_message


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: Any, path: str | Path) -> None:
    """Save JSON data with deterministic pretty printing."""
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def load_json(path: str | Path) -> Any:
    """Load JSON data from disk."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def log_progress(message: str, verbose: int, min_verbose: int = 1) -> None:
    """Print a concise progress message when verbosity is high enough."""
    log_message(message, verbose, min_verbose)


def save_experiment_logs(
    group_name: str,
    training_metrics: dict[str, Any],
    system_metrics: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    """Save standard experiment log files under `logs/<group_name>/`.
    """
    log_dir = ensure_dir(Path("logs") / group_name)
    save_json(training_metrics, log_dir / "training_metrics.json")
    save_json(system_metrics, log_dir / "system_metrics.json")
    save_json(summary, log_dir / "experiment_summary.json")
