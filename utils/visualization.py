"""Visualization helpers for Grid-world experiments."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns

from envs.grid_world import State

COOL_LINE_COLOR = "#2A9DCC"
COOL_GRID_COLOR = "#D8EEF2"
COOL_TEXT_COLOR = "#1F3A4A"
COOL_BAR_PALETTE = ["#7CC7E8", "#79D6C9", "#A7E8BD", "#8AB6F9", "#9ADBC5"]
COOL_HEATMAP_CMAP = sns.color_palette("crest", as_cmap=True)
GRID_LAYOUT_COLORS = {
    "normal": "#F8FCFD",
    "start": "#7CC7E8",
    "goal": "#8DE0B7",
    "trap": "#F2A08B",
    "wall": "#556B78",
    "edge": COOL_GRID_COLOR,
    "text": COOL_TEXT_COLOR,
    "light_text": "white",
}

sns.set_theme(
    context="notebook",
    style="whitegrid",
    palette=COOL_BAR_PALETTE,
    rc={
        "axes.edgecolor": COOL_GRID_COLOR,
        "axes.labelcolor": COOL_TEXT_COLOR,
        "axes.titlecolor": COOL_TEXT_COLOR,
        "grid.color": COOL_GRID_COLOR,
        "figure.facecolor": "white",
        "axes.facecolor": "#F8FCFD",
    },
)


def plot_grid_world_layout(
    env: Any,
    title: str = "Grid-world 8x8 Layout",
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Any, Any]:
    """Plot the static Grid-world layout with start, goal, trap, and wall cells."""
    if save_path is not None:
        _ensure_parent(save_path)

    rows, cols = env.grid_size
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    axis.set_facecolor(GRID_LAYOUT_COLORS["normal"])

    for row in range(rows):
        for col in range(cols):
            state = (row, col)
            fill_color = GRID_LAYOUT_COLORS["normal"]
            label = ""
            text_color = GRID_LAYOUT_COLORS["text"]

            if state == env.start_state:
                fill_color = GRID_LAYOUT_COLORS["start"]
                label = "S"
                text_color = GRID_LAYOUT_COLORS["light_text"]
            elif state in env.goal_states:
                fill_color = GRID_LAYOUT_COLORS["goal"]
                label = "G"
                text_color = GRID_LAYOUT_COLORS["text"]
            elif state in env.trap_states:
                fill_color = GRID_LAYOUT_COLORS["trap"]
                label = "T"
                text_color = GRID_LAYOUT_COLORS["text"]
            elif state in env.wall_states:
                fill_color = GRID_LAYOUT_COLORS["wall"]
                label = "W"
                text_color = GRID_LAYOUT_COLORS["light_text"]

            axis.add_patch(
                Rectangle(
                    (col, row),
                    1,
                    1,
                    facecolor=fill_color,
                    edgecolor=GRID_LAYOUT_COLORS["edge"],
                    linewidth=0.9,
                )
            )
            if label:
                axis.text(
                    col + 0.5,
                    row + 0.5,
                    label,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=13,
                    fontweight="bold",
                )

    axis.set_title(title)
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")
    axis.set_xlim(0, cols)
    axis.set_ylim(rows, 0)
    axis.set_aspect("equal")
    axis.set_xticks([col + 0.5 for col in range(cols)])
    axis.set_xticklabels(range(cols), rotation=0)
    axis.set_yticks([row + 0.5 for row in range(rows)])
    axis.set_yticklabels(range(rows), rotation=0)
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)

    _finalize_figure(figure, save_path, show)
    return figure, axis


def plot_value_heatmap(
    values: dict[State, float],
    env: Any,
    title: str,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Any, Any]:
    """Plot a value-function heatmap and save it to disk."""
    if save_path is not None:
        _ensure_parent(save_path)
    grid = np.full(env.grid_size, np.nan)
    for state, value in values.items():
        grid[state] = value

    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(
        grid,
        ax=axis,
        cmap=COOL_HEATMAP_CMAP,
        linewidths=0.6,
        linecolor="#E3F4F5",
        square=True,
        cbar_kws={"label": "Value"},
    )
    axis.set_title(title)
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")
    axis.set_xticklabels(range(env.grid_size[1]), rotation=0)
    axis.set_yticklabels(range(env.grid_size[0]), rotation=0)
    _annotate_special_states(axis, env)
    _finalize_figure(figure, save_path, show)
    return figure, axis


def plot_policy_arrows(
    policy: dict[State, str],
    env: Any,
    title: str,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Any, Any]:
    """Plot a simple policy-arrow grid and save it to disk."""
    if save_path is not None:
        _ensure_parent(save_path)
    arrow_map = {"up": "^", "down": "v", "left": "<", "right": ">"}
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    background = np.zeros(env.grid_size)
    sns.heatmap(
        background,
        ax=axis,
        cmap=sns.light_palette("#7CC7E8", as_cmap=True),
        cbar=False,
        linewidths=0.6,
        linecolor="#D8EEF2",
        square=True,
    )
    axis.set_title(title)
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")
    axis.set_xticklabels(range(env.grid_size[1]), rotation=0)
    axis.set_yticklabels(range(env.grid_size[0]), rotation=0)
    for state, action in policy.items():
        axis.text(
            state[1] + 0.5,
            state[0] + 0.5,
            arrow_map.get(action, "?"),
            ha="center",
            va="center",
            color="#134E5E",
            fontsize=13,
            fontweight="bold",
        )
    _annotate_special_states(axis, env)
    _finalize_figure(figure, save_path, show)
    return figure, axis


def plot_learning_curve(episode_returns: list[float], title: str, save_path: str | Path) -> None:
    """Plot episodic returns over training episodes."""
    _plot_line(episode_returns, title, "Episode", "Return", save_path)


def plot_moving_average(values: list[float], title: str, save_path: str | Path) -> None:
    """Plot a moving-average learning curve."""
    _plot_line(values, title, "Episode", "Moving average return", save_path)


def plot_episode_steps(episode_steps: list[int], title: str, save_path: str | Path) -> None:
    """Plot episode length over training episodes."""
    _plot_line(episode_steps, title, "Episode", "Steps", save_path)


def plot_td_error_curve(td_errors: list[float], title: str, save_path: str | Path) -> None:
    """Plot absolute TD error values over updates."""
    _plot_line(
        [abs(error) for error in td_errors],
        title,
        "Update",
        "Absolute TD error",
        save_path,
    )


def plot_td_mse_curve(
    episodes: list[int],
    mse_values: list[float],
    title: str,
    save_path: str | Path,
) -> None:
    """Plot TD prediction MSE against a Policy Evaluation baseline."""
    _plot_xy_line(episodes, mse_values, title, "Episode", "MSE vs PE", save_path)


def plot_success_trap_curves(
    episodes: list[int],
    success_rates: list[float],
    trap_rates: list[float],
    title: str,
    save_path: str | Path,
) -> None:
    """Plot trailing-window success and trap rates over episodes."""
    _ensure_parent(save_path)
    _, axis = plt.subplots(figsize=(7.0, 4.4))
    sns.lineplot(
        x=episodes,
        y=success_rates,
        ax=axis,
        label="success",
        color="#2A9D8F",
        linewidth=2.2,
    )
    sns.lineplot(
        x=episodes,
        y=trap_rates,
        ax=axis,
        label="trap",
        color="#E76F51",
        linewidth=2.2,
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_title(title)
    axis.set_xlabel("Episode")
    axis.set_ylabel("Trailing-window rate")
    axis.legend()
    _polish_axis(axis)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_sensitivity_bar_or_line(
    x_values: list[float],
    y_values: list[float],
    title: str,
    save_path: str | Path,
    xlabel: str,
    ylabel: str,
) -> None:
    """Plot a simple sensitivity curve for one scalar metric."""
    _plot_xy_line(x_values, y_values, title, xlabel, ylabel, save_path)


def plot_success_trap_rates(
    rates: dict[str, float],
    title: str,
    save_path: str | Path,
) -> None:
    """Plot final success/trap/timeout rates for a learner."""
    _ensure_parent(save_path)
    _, axis = plt.subplots(figsize=(6.2, 4.2))
    sns.barplot(
        x=list(rates.keys()),
        y=list(rates.values()),
        ax=axis,
        palette=["#8DE0B7", "#7CC7E8", "#AEB8C2"],
        hue=list(rates.keys()),
        legend=False,
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_title(title)
    axis.set_ylabel("Rate")
    _polish_axis(axis)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_bellman_residual(
    residuals: list[float],
    title: str,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Any, Any]:
    """Plot Bellman residual values over iterations."""
    return _plot_line(residuals, title, "Iteration", "Bellman residual", save_path, show)


def plot_policy_changes(
    policy_changes: Sequence[int],
    title: str = "Policy Iteration - Policy Changes per Iteration",
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Any, Any]:
    """Plot changed-action counts over policy improvement iterations."""
    if save_path is not None:
        _ensure_parent(save_path)

    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    iterations = list(range(1, len(policy_changes) + 1))
    sns.lineplot(
        x=iterations,
        y=policy_changes,
        ax=axis,
        color=COOL_LINE_COLOR,
        linewidth=2.2,
        marker="o",
    )
    axis.set_title(title)
    axis.set_xlabel("Policy improvement iteration")
    axis.set_ylabel("Number of changed states")
    if iterations:
        axis.set_xticks(iterations)
    _polish_axis(axis)

    _finalize_figure(figure, save_path, show)
    return figure, axis


def plot_comparison_bar(
    metrics: dict[str, float],
    title: str,
    save_path: str | Path | None = None,
    ylabel: str = "Value",
    show: bool = False,
) -> tuple[Any, Any]:
    """Plot a bar chart comparing scalar metrics across algorithms."""
    if save_path is not None:
        _ensure_parent(save_path)
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    labels = list(metrics.keys())
    sns.barplot(
        x=labels,
        y=list(metrics.values()),
        ax=axis,
        palette=sns.color_palette(COOL_BAR_PALETTE, n_colors=len(labels)),
        hue=labels,
        legend=False,
    )
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=30)
    _polish_axis(axis)
    _finalize_figure(figure, save_path, show)
    return figure, axis


def _plot_line(
    values: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str | Path | None,
    show: bool = False,
) -> tuple[Any, Any]:
    """Plot a generic line chart."""
    if save_path is not None:
        _ensure_parent(save_path)
    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    sns.lineplot(
        x=list(range(len(values))),
        y=values,
        ax=axis,
        color=COOL_LINE_COLOR,
        linewidth=2.2,
    )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    _polish_axis(axis)
    _finalize_figure(figure, save_path, show)
    return figure, axis


def _plot_xy_line(
    x_values: list[int] | list[float],
    y_values: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str | Path,
) -> None:
    """Plot a generic line chart with explicit x values."""
    _ensure_parent(save_path)
    _, axis = plt.subplots(figsize=(7.0, 4.4))
    sns.lineplot(
        x=x_values,
        y=y_values,
        ax=axis,
        color=COOL_LINE_COLOR,
        linewidth=2.2,
        marker="o",
    )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    _polish_axis(axis)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _annotate_special_states(axis: Any, env: Any) -> None:
    """Add Start/Goal/Trap/Wall labels to a grid plot."""
    axis.text(
        env.start_state[1] + 0.5,
        env.start_state[0] + 0.5,
        "S",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )
    for state in env.goal_states:
        axis.text(
            state[1] + 0.5,
            state[0] + 0.5,
            "G",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
    for state in env.trap_states:
        axis.text(
            state[1] + 0.5,
            state[0] + 0.5,
            "T",
            ha="center",
            va="center",
            color="red",
            fontweight="bold",
        )
    for state in env.wall_states:
        axis.text(
            state[1] + 0.5,
            state[0] + 0.5,
            "#",
            ha="center",
            va="center",
            color="black",
            fontweight="bold",
        )


def _polish_axis(axis: Any) -> None:
    """Apply consistent cool-toned finishing touches."""
    axis.grid(True, axis="y", alpha=0.35)
    sns.despine(ax=axis, left=False, bottom=False)


def _ensure_parent(save_path: str | Path) -> None:
    """Create the parent directory for an output figure."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)


def _finalize_figure(figure: Any, save_path: str | Path | None, show: bool) -> None:
    """Save and optionally display a Matplotlib figure."""
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
