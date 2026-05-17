"""Metric helpers for Grid-world experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

NumberMap = Mapping[Any, float]


def mean_squared_error(values_a: NumberMap, values_b: NumberMap) -> float:
    """Compute mean squared error over keys shared by two value mappings."""
    keys = _shared_keys(values_a, values_b)
    if not keys:
        return 0.0
    return sum((values_a[key] - values_b[key]) ** 2 for key in keys) / len(keys)


def mean_absolute_error(values_a: NumberMap, values_b: NumberMap) -> float:
    """Compute mean absolute error over keys shared by two value mappings."""
    keys = _shared_keys(values_a, values_b)
    if not keys:
        return 0.0
    return sum(abs(values_a[key] - values_b[key]) for key in keys) / len(keys)


def max_abs_error(values_a: NumberMap, values_b: NumberMap) -> float:
    """Compute maximum absolute error over keys shared by two value mappings."""
    keys = _shared_keys(values_a, values_b)
    if not keys:
        return 0.0
    return max(abs(values_a[key] - values_b[key]) for key in keys)


def policy_agreement(policy_a: Mapping[Any, Any], policy_b: Mapping[Any, Any], states: Iterable[Any]) -> float:
    """Compute the fraction of states where two policies select the same action."""
    state_list = list(states)
    if not state_list:
        return 0.0
    matches = sum(1 for state in state_list if policy_a.get(state) == policy_b.get(state))
    return matches / len(state_list)


def compute_success_rate(episodes: Sequence[Mapping[str, Any]]) -> float:
    """Compute the fraction of episodes that reached a goal state."""
    if not episodes:
        return 0.0
    successes = sum(1 for episode in episodes if episode.get("outcome") == "success")
    return successes / len(episodes)


def compute_trap_rate(episodes: Sequence[Mapping[str, Any]]) -> float:
    """Compute the fraction of episodes that ended in a trap state."""
    if not episodes:
        return 0.0
    traps = sum(1 for episode in episodes if episode.get("outcome") == "trap")
    return traps / len(episodes)


def compute_average_return(episode_returns: Sequence[float]) -> float:
    """Compute average episodic return."""
    if not episode_returns:
        return 0.0
    return sum(episode_returns) / len(episode_returns)


def compute_average_steps(episode_steps: Sequence[int]) -> float:
    """Compute average number of steps per episode."""
    if not episode_steps:
        return 0.0
    return sum(episode_steps) / len(episode_steps)


def compute_bellman_residual(old_values: NumberMap, new_values: NumberMap) -> float:
    """Compute max-norm difference between two value-function iterates."""
    return max_abs_error(old_values, new_values)


def _shared_keys(values_a: NumberMap, values_b: NumberMap) -> list[Any]:
    """Return keys present in both mappings."""
    return [key for key in values_a if key in values_b]
