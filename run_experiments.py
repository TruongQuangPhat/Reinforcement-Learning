"""Experiment entry point for the 8x8 Grid-world RL project."""

from __future__ import annotations

from typing import Any

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


def run_planning_experiments() -> dict[str, Any]:
    """Run planning experiment skeletons on the default 8x8 Grid-world."""
    env = PlanningGridWorld(grid_size=(8, 8), seed=42)
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
    """Run learning experiment skeletons on the default 8x8 Grid-world."""
    env = LearningGridWorld(grid_size=(8, 8), seed=42)
    learners = {
        "td_zero": TDZero(env),
        "td_lambda": TDLambda(env),
        "sarsa": SARSA(env),
        "q_learning": QLearning(env),
    }

    training_metrics: dict[str, Any] = {}
    system_metrics: dict[str, Any] = {}
    for name, learner in learners.items():
        profile = profile_callable(learner.train)
        training_metrics[name] = learner.get_metrics()
        system_metrics[name] = {
            "wall_time_sec": profile.wall_time_sec,
            "cpu_time_sec": profile.cpu_time_sec,
            "current_memory_mb": profile.current_memory_mb,
            "peak_memory_mb": profile.peak_memory_mb,
        }

    summary = {
        "group": "learning",
        "status": "skeleton_only",
        "algorithms": list(learners.keys()),
        "grid_size": env.grid_size,
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


if __name__ == "__main__":
    main()
