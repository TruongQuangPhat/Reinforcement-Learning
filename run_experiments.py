"""Experiment entry point for the 8x8 Grid-world RL project."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from agents.learning import QLearning, SARSA, TDLambda, TDZero
from agents.planning import (
    LinearProgrammingPlanner,
    PolicyEvaluation,
    PolicyIteration,
    ValueIteration,
)
from envs.learning_grid_world import LearningGridWorld
from envs.planning_grid_world import PlanningGridWorld
from utils.experiment_io import save_experiment_logs
from utils.profiling import profile_callable
from utils.visualization import (
    plot_episode_steps,
    plot_learning_curve,
    plot_moving_average,
    plot_policy_arrows,
    plot_success_trap_rates,
    plot_td_error_curve,
    plot_value_heatmap,
)


RANDOM_SEED = 42
LEARNING_EPISODES = 1_000
LEARNING_MAX_STEPS = 256
FIGURE_DIR = Path("report") / "figures" / "learning"


def run_planning_experiments() -> dict[str, Any]:
    """Run planning experiment skeletons on the default 8x8 Grid-world."""
    env = PlanningGridWorld(grid_size=(8, 8), seed=RANDOM_SEED)
    planners = {
        "policy_evaluation": PolicyEvaluation(env),
        "policy_iteration": PolicyIteration(env),
        "value_iteration": ValueIteration(env),
        "linear_programming": LinearProgrammingPlanner(env),
    }

    training_metrics: dict[str, Any] = {}
    system_metrics: dict[str, Any] = {}
    for name, planner in planners.items():
        profile = profile_callable(planner.run)
        training_metrics[name] = planner.get_metrics()
        system_metrics[name] = {
            "wall_time_sec": profile.wall_time_sec,
            "cpu_time_sec": profile.cpu_time_sec,
            "current_memory_mb": profile.current_memory_mb,
            "peak_memory_mb": profile.peak_memory_mb,
        }

    summary = {
        "group": "planning",
        "status": "skeleton_only",
        "algorithms": list(planners.keys()),
        "grid_size": env.grid_size,
    }
    save_experiment_logs("planning", training_metrics, system_metrics, summary)
    return summary


def run_learning_experiments() -> dict[str, Any]:
    """Run model-free learning experiments on the default 8x8 Grid-world."""
    env = LearningGridWorld(grid_size=(8, 8), seed=RANDOM_SEED)
    learners = {
        "td_zero": TDZero(
            LearningGridWorld(grid_size=(8, 8), seed=RANDOM_SEED),
            episodes=LEARNING_EPISODES,
            max_steps_per_episode=LEARNING_MAX_STEPS,
            seed=RANDOM_SEED,
        ),
        "td_lambda": TDLambda(
            LearningGridWorld(grid_size=(8, 8), seed=RANDOM_SEED),
            episodes=LEARNING_EPISODES,
            max_steps_per_episode=LEARNING_MAX_STEPS,
            seed=RANDOM_SEED,
        ),
        "sarsa": SARSA(
            LearningGridWorld(grid_size=(8, 8), seed=RANDOM_SEED),
            episodes=LEARNING_EPISODES,
            max_steps_per_episode=LEARNING_MAX_STEPS,
            seed=RANDOM_SEED,
        ),
        "q_learning": QLearning(
            LearningGridWorld(grid_size=(8, 8), seed=RANDOM_SEED),
            episodes=LEARNING_EPISODES,
            max_steps_per_episode=LEARNING_MAX_STEPS,
            seed=RANDOM_SEED,
        ),
    }

    training_metrics: dict[str, Any] = {}
    system_metrics: dict[str, Any] = {}
    for name, learner in learners.items():
        learner.train()
        training_metrics[name] = learner.get_metrics()
        _save_learning_figures(name, learner, training_metrics[name])
        system_metrics[name] = {
            "wall_time_sec": training_metrics[name].get("runtime_sec", 0.0),
            "cpu_time_sec": training_metrics[name].get("cpu_time_sec", 0.0),
            "current_memory_mb": training_metrics[name].get("current_memory_mb", 0.0),
            "peak_memory_mb": training_metrics[name].get("peak_memory_mb", 0.0),
            "environment_steps": training_metrics[name].get("environment_steps", 0),
            "q_updates": training_metrics[name].get("q_updates", 0),
        }

    summary = {
        "group": "learning",
        "status": "completed",
        "algorithms": list(learners.keys()),
        "best_control_by_average_return": _best_by_metric(training_metrics, "average_return"),
        "lowest_td_prediction_error": _lowest_final_td_error(training_metrics),
        "figure_dir": str(FIGURE_DIR),
        "environment": _environment_summary(env),
        "metadata": _experiment_metadata(),
    }
    save_experiment_logs("learning", training_metrics, system_metrics, summary)
    return summary


def run_comparison_experiments() -> dict[str, Any]:
    """Prepare cross-group comparison outputs.

    TODO: Compare convergence speed, final return, policy quality, and resource
    usage after planning and learning algorithms are implemented.
    """
    return {
        "status": "not_implemented",
        "description": "Comparison will be added after algorithm implementations.",
    }


def main() -> None:
    """Run all experiment groups."""
    planning_summary = run_planning_experiments()
    learning_summary = run_learning_experiments()
    comparison_summary = run_comparison_experiments()
    print(
        {
            "planning": planning_summary["status"],
            "learning": learning_summary["status"],
            "comparison": comparison_summary["status"],
        }
    )


def _save_learning_figures(name: str, learner: Any, metrics: dict[str, Any]) -> None:
    """Save standard learning figures for one trained learner."""
    env = learner.env
    if "episode_returns" in metrics:
        plot_learning_curve(
            metrics["episode_returns"],
            f"{name}: episode returns",
            FIGURE_DIR / f"{name}_episode_returns.png",
        )
    if "moving_average_returns" in metrics:
        plot_moving_average(
            metrics["moving_average_returns"],
            f"{name}: moving average returns",
            FIGURE_DIR / f"{name}_moving_average_returns.png",
        )
    if "episode_steps" in metrics:
        plot_episode_steps(
            metrics["episode_steps"],
            f"{name}: steps per episode",
            FIGURE_DIR / f"{name}_episode_steps.png",
        )
    if "td_errors" in metrics:
        plot_td_error_curve(
            metrics["td_errors"],
            f"{name}: TD errors",
            FIGURE_DIR / f"{name}_td_errors.png",
        )
    if "mean_absolute_td_error_per_episode" in metrics:
        plot_td_error_curve(
            metrics["mean_absolute_td_error_per_episode"],
            f"{name}: mean absolute TD error",
            FIGURE_DIR / f"{name}_mean_abs_td_error.png",
        )
    if "success_rate" in metrics and "trap_rate" in metrics:
        timeout_rate = max(0.0, 1.0 - metrics["success_rate"] - metrics["trap_rate"])
        plot_success_trap_rates(
            {
                "success": metrics["success_rate"],
                "trap": metrics["trap_rate"],
                "timeout": timeout_rate,
            },
            f"{name}: final outcome rates",
            FIGURE_DIR / f"{name}_outcome_rates.png",
        )
    if hasattr(learner, "get_value_function"):
        plot_value_heatmap(
            learner.get_value_function(),
            env,
            f"{name}: value function",
            FIGURE_DIR / f"{name}_value_heatmap.png",
        )
    policy = learner.get_policy()
    if policy:
        plot_policy_arrows(
            policy,
            env,
            f"{name}: learned policy",
            FIGURE_DIR / f"{name}_policy.png",
        )


def _environment_summary(env: LearningGridWorld) -> dict[str, Any]:
    """Return JSON-safe environment configuration metadata."""
    return {
        "random_seed": env.seed,
        "grid_size": list(env.grid_size),
        "start_state": list(env.start_state),
        "goal_states": [list(state) for state in sorted(env.goal_states)],
        "trap_states": [list(state) for state in sorted(env.trap_states)],
        "wall_states": [list(state) for state in sorted(env.wall_states)],
        "reward_config": env.reward_config,
        "gamma": env.gamma,
    }


def _experiment_metadata() -> dict[str, Any]:
    """Return reproducibility metadata for experiment logs."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }


def _best_by_metric(metrics_by_algorithm: dict[str, dict[str, Any]], metric: str) -> str | None:
    """Return the algorithm with the largest scalar metric."""
    candidates = {
        name: metrics[metric]
        for name, metrics in metrics_by_algorithm.items()
        if isinstance(metrics.get(metric), (int, float))
    }
    if not candidates:
        return None
    return max(candidates, key=candidates.get)


def _lowest_final_td_error(metrics_by_algorithm: dict[str, dict[str, Any]]) -> str | None:
    """Return the prediction learner with the lowest final TD error."""
    candidates = {
        name: metrics["final_mean_absolute_td_error"]
        for name, metrics in metrics_by_algorithm.items()
        if name in {"td_zero", "td_lambda"} and "final_mean_absolute_td_error" in metrics
    }
    if not candidates:
        return None
    return min(candidates, key=candidates.get)


if __name__ == "__main__":
    main()
