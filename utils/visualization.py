"""Visualization helpers for Grid-world experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from envs.grid_world import State


def plot_value_heatmap(values: dict[State, float], env: Any, title: str, save_path: str | Path) -> None:
    """Plot a value-function heatmap and save it to disk."""
    _ensure_parent(save_path)
    grid = np.full(env.grid_size, np.nan)
    for state, value in values.items():
        grid[state] = value

    _, axis = plt.subplots()
    image = axis.imshow(grid, cmap="viridis")
    axis.set_title(title)
    plt.colorbar(image, ax=axis)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_policy_arrows(policy: dict[State, str], env: Any, title: str, save_path: str | Path) -> None:
    """Plot a simple policy-arrow grid and save it to disk."""
    _ensure_parent(save_path)
    arrow_map = {"up": "^", "down": "v", "left": "<", "right": ">"}
    _, axis = plt.subplots()
    axis.set_title(title)
    axis.set_xlim(-0.5, env.grid_size[1] - 0.5)
    axis.set_ylim(env.grid_size[0] - 0.5, -0.5)
    axis.set_xticks(range(env.grid_size[1]))
    axis.set_yticks(range(env.grid_size[0]))
    axis.grid(True)
    for state, action in policy.items():
        axis.text(state[1], state[0], arrow_map.get(action, "?"), ha="center", va="center")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_learning_curve(episode_returns: list[float], title: str, save_path: str | Path) -> None:
    """Plot episodic returns over training episodes."""
    _plot_line(episode_returns, title, "Episode", "Return", save_path)


def plot_bellman_residual(residuals: list[float], title: str, save_path: str | Path) -> None:
    """Plot Bellman residual values over iterations."""
    _plot_line(residuals, title, "Iteration", "Bellman residual", save_path)


def plot_comparison_bar(metrics: dict[str, float], title: str, save_path: str | Path) -> None:
    """Plot a bar chart comparing scalar metrics across algorithms."""
    _ensure_parent(save_path)
    _, axis = plt.subplots()
    axis.bar(list(metrics.keys()), list(metrics.values()))
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=30)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _plot_line(values: list[float], title: str, xlabel: str, ylabel: str, save_path: str | Path) -> None:
    """Plot a generic line chart."""
    _ensure_parent(save_path)
    _, axis = plt.subplots()
    axis.plot(values)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _ensure_parent(save_path: str | Path) -> None:
    """Create the parent directory for an output figure."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
