"""Experiment entry point for the 8x8 Grid-world RL project."""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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
from utils.metrics import max_abs_error, policy_agreement
from utils.profiling import profile_callable
from utils.visualization import (
    plot_bellman_residual,
    plot_comparison_bar,
    plot_policy_arrows,
    plot_value_heatmap,
)


def run_planning_experiments() -> dict[str, Any]:
    """Run full planning experiments on the default 8x8 Grid-world."""
    env = PlanningGridWorld(grid_size=(8, 8), seed=42)
    fixed_policy = _build_goal_directed_policy(env)
    planners = {
        "policy_evaluation": PolicyEvaluation(env, policy=fixed_policy),
        "policy_iteration": PolicyIteration(env),
        "value_iteration": ValueIteration(env),
        "linear_programming": LinearProgrammingPlanner(env),
    }

    training_metrics: dict[str, Any] = {}
    system_metrics: dict[str, Any] = {}
    for name, planner in planners.items():
        profile = profile_callable(planner.run)
        training_metrics[name] = planner.get_metrics()
        training_metrics[name]["algorithm_name"] = name
        training_metrics[name]["hyperparameters"] = _planner_hyperparameters(planner)
        system_metrics[name] = {
            "wall_time_sec": profile.wall_time_sec,
            "cpu_time_sec": profile.cpu_time_sec,
            "current_memory_mb": profile.current_memory_mb,
            "peak_memory_mb": profile.peak_memory_mb,
        }
        training_metrics[name].update(system_metrics[name])

    _add_planning_comparisons(env, planners, training_metrics, system_metrics)
    figure_paths = _save_planning_figures(env, planners, system_metrics)

    summary = {
        "group": "planning",
        "status": "completed",
        "algorithms": list(planners.keys()),
        "metadata": _experiment_metadata(env),
        "comparison_scope": (
            "Policy Evaluation evaluates a fixed heuristic policy. "
            "Policy Iteration, Value Iteration, and Linear Programming solve "
            "for an optimal policy using the model."
        ),
        "fastest_fixed_policy_evaluation": "policy_evaluation",
        "best_optimal_control_runtime_algorithm": min(
            ("policy_iteration", "value_iteration", "linear_programming"),
            key=lambda algorithm: system_metrics[algorithm]["wall_time_sec"],
        ),
        "pi_vi_policy_agreement": training_metrics["policy_iteration"][
            "policy_agreement_vs_value_iteration"
        ],
        "lp_vi_max_abs_error": training_metrics["linear_programming"][
            "value_error_vs_value_iteration"
        ],
        "figure_paths": figure_paths,
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


def _build_goal_directed_policy(env: PlanningGridWorld) -> dict[tuple[int, int], str]:
    """Build a deterministic fixed policy for policy evaluation."""
    policy: dict[tuple[int, int], str] = {}
    goals = list(env.goal_states)
    for state in env.get_states():
        if env.is_terminal(state):
            continue

        best_action = env.get_actions(state)[0]
        best_distance = float("inf")
        best_terminal_penalty = 0
        for action in env.get_actions(state):
            transitions = env.get_transitions(state, action)
            _, next_state, _, _ = transitions[0]
            distance = min(
                abs(next_state[0] - goal[0]) + abs(next_state[1] - goal[1])
                for goal in goals
            )
            terminal_penalty = 1 if next_state in env.trap_states else 0
            candidate = (terminal_penalty, distance)
            current = (best_terminal_penalty, best_distance)
            if candidate < current:
                best_action = action
                best_distance = distance
                best_terminal_penalty = terminal_penalty
        policy[state] = best_action
    return policy


def _planner_hyperparameters(planner: Any) -> dict[str, Any]:
    """Extract common planner hyperparameters for logs."""
    hyperparameters: dict[str, Any] = {}
    if hasattr(planner, "theta"):
        hyperparameters["theta"] = planner.theta
    if hasattr(planner, "max_iterations"):
        hyperparameters["max_iterations"] = planner.max_iterations
    if hasattr(planner, "solver_options"):
        hyperparameters["solver_options"] = planner.solver_options
    return hyperparameters


def _experiment_metadata(env: PlanningGridWorld) -> dict[str, Any]:
    """Return reproducibility metadata for one experiment run."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "random_seed": env.seed,
        "grid_size": list(env.grid_size),
        "start_state": list(env.start_state),
        "goal_states": _serialize_states(env.goal_states),
        "trap_states": _serialize_states(env.trap_states),
        "wall_states": _serialize_states(env.wall_states),
        "reward_config": env.reward_config,
        "gamma": env.gamma,
    }


def _serialize_states(states: set[tuple[int, int]]) -> list[list[int]]:
    """Serialize a state set in stable row-major order."""
    return [list(state) for state in sorted(states)]


def _add_planning_comparisons(
    env: PlanningGridWorld,
    planners: dict[str, Any],
    training_metrics: dict[str, Any],
    system_metrics: dict[str, Any],
) -> None:
    """Add PI/VI/LP comparison metrics to training logs."""
    vi_values = planners["value_iteration"].get_value_function()
    vi_policy = planners["value_iteration"].get_policy()
    comparison_states = [
        state for state in env.get_states() if not env.is_terminal(state)
    ]

    for name in ("policy_iteration", "linear_programming"):
        values = planners[name].get_value_function()
        policy = planners[name].get_policy()
        training_metrics[name]["value_error_vs_value_iteration"] = max_abs_error(
            values,
            vi_values,
        )
        training_metrics[name]["policy_agreement_vs_value_iteration"] = (
            policy_agreement(policy, vi_policy, comparison_states)
        )

    runtime_comparison = {
        name: metrics["wall_time_sec"]
        for name, metrics in system_metrics.items()
    }
    backup_comparison = {
        name: metrics.get("bellman_backups", 0)
        for name, metrics in training_metrics.items()
    }
    training_metrics["planning_comparison"] = {
        "runtime_sec": runtime_comparison,
        "bellman_backups": backup_comparison,
        "policy_iteration_vs_value_iteration_max_abs_error": (
            training_metrics["policy_iteration"]["value_error_vs_value_iteration"]
        ),
        "linear_programming_vs_value_iteration_max_abs_error": (
            training_metrics["linear_programming"]["value_error_vs_value_iteration"]
        ),
        "policy_iteration_vs_value_iteration_policy_agreement": (
            training_metrics["policy_iteration"]["policy_agreement_vs_value_iteration"]
        ),
        "linear_programming_vs_value_iteration_policy_agreement": (
            training_metrics["linear_programming"]["policy_agreement_vs_value_iteration"]
        ),
    }


def _save_planning_figures(
    env: PlanningGridWorld,
    planners: dict[str, Any],
    system_metrics: dict[str, Any],
) -> list[str]:
    """Save standard planning figures for the report."""
    figure_paths = [
        "report/figures/planning/policy_evaluation_values.png",
        "report/figures/planning/fixed_policy_arrows.png",
        "report/figures/planning/value_iteration_values.png",
        "report/figures/planning/value_iteration_policy.png",
        "report/figures/planning/value_iteration_residuals.png",
        "report/figures/planning/policy_iteration_policy_changes.png",
        "report/figures/planning/planning_runtime_comparison.png",
    ]

    plot_value_heatmap(
        planners["policy_evaluation"].get_value_function(),
        env,
        "Policy Evaluation: V^pi",
        figure_paths[0],
    )
    plot_policy_arrows(
        planners["policy_evaluation"].get_policy(),
        env,
        "Fixed Policy Used for Policy Evaluation",
        figure_paths[1],
    )
    plot_value_heatmap(
        planners["value_iteration"].get_value_function(),
        env,
        "Value Iteration: V*",
        figure_paths[2],
    )
    plot_policy_arrows(
        planners["value_iteration"].get_policy(),
        env,
        "Value Iteration Greedy Policy",
        figure_paths[3],
    )
    plot_bellman_residual(
        planners["value_iteration"].get_metrics()["bellman_residuals"],
        "Value Iteration Bellman Residual",
        figure_paths[4],
    )
    plot_bellman_residual(
        planners["policy_iteration"].get_metrics()["policy_changes_per_iteration"],
        "Policy Iteration Policy Changes",
        figure_paths[5],
    )
    plot_comparison_bar(
        {
            name: metrics["wall_time_sec"]
            for name, metrics in system_metrics.items()
        },
        "Planning Runtime Comparison",
        figure_paths[6],
    )
    return figure_paths


if __name__ == "__main__":
    main()
