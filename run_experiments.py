"""Experiment entry point for the 8x8 Grid-world RL project."""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
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
from envs.grid_world import State
from envs.learning_grid_world import LearningGridWorld
from envs.planning_grid_world import PlanningGridWorld
from utils.experiment_io import load_json, save_experiment_logs
from utils.metrics import max_abs_error, mean_squared_error, policy_agreement
from utils.profiling import profile_callable
from utils.visualization import (
    plot_bellman_residual,
    plot_comparison_bar,
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
PLANNING_FIGURE_DIR = Path("report") / "figures" / "planning"
LEARNING_FIGURE_DIR = Path("report") / "figures" / "learning"
COMPARISON_FIGURE_DIR = Path("report") / "figures" / "comparison"


def run_planning_experiments() -> dict[str, Any]:
    """Run full planning experiments on the default 8x8 Grid-world."""
    env = PlanningGridWorld(grid_size=(8, 8), seed=RANDOM_SEED)
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
        training_metrics[name]["hyperparameters"] = _algorithm_hyperparameters(planner)
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
        training_metrics[name]["algorithm_name"] = name
        training_metrics[name]["hyperparameters"] = _algorithm_hyperparameters(learner)
        _save_learning_figures(name, learner, training_metrics[name])
        system_metrics[name] = {
            "wall_time_sec": training_metrics[name].get("runtime_sec", 0.0),
            "cpu_time_sec": training_metrics[name].get("cpu_time_sec", 0.0),
            "current_memory_mb": training_metrics[name].get("current_memory_mb", 0.0),
            "peak_memory_mb": training_metrics[name].get("peak_memory_mb", 0.0),
            "environment_steps": training_metrics[name].get("environment_steps", 0),
            "q_updates": training_metrics[name].get("q_updates", 0),
        }

    baseline_info = _add_learning_planning_baselines(learners, training_metrics)
    learning_comparison_paths = _save_learning_comparison_figures(training_metrics)
    summary = {
        "group": "learning",
        "status": "completed",
        "algorithms": list(learners.keys()),
        "metadata": _experiment_metadata(env),
        "comparison_scope": (
            "TD(0) and TD(lambda) are compared against Policy Evaluation. "
            "SARSA and Q-learning are compared against Value Iteration."
        ),
        "best_control_by_average_return": _best_by_metric(
            training_metrics,
            "average_return",
        ),
        "lowest_td_prediction_error": _lowest_final_td_error(training_metrics),
        "planning_baselines": baseline_info,
        "figure_dir": str(LEARNING_FIGURE_DIR),
        "comparison_figure_paths": learning_comparison_paths,
    }
    save_experiment_logs("learning", training_metrics, system_metrics, summary)
    return summary


def run_comparison_experiments() -> dict[str, Any]:
    """Prepare report figures for valid cross-group comparisons."""
    figure_paths = _save_valid_comparison_figures()
    return {
        "status": "completed",
        "description": (
            "Planning algorithms use the full model. Learning algorithms use "
            "sampled environment interactions and are compared only against "
            "matching planning baselines."
        ),
        "figure_paths": figure_paths,
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


def _build_goal_directed_policy(env: PlanningGridWorld) -> dict[State, str]:
    """Build a deterministic fixed policy for policy evaluation."""
    policy: dict[State, str] = {}
    goals = list(env.goal_states)
    for state in env.get_states():
        if env.is_terminal(state):
            continue

        best_action = env.get_actions(state)[0]
        best_distance = float("inf")
        best_terminal_penalty = 0
        for action in env.get_actions(state):
            _, next_state, _, _ = env.get_transitions(state, action)[0]
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


def _algorithm_hyperparameters(algorithm: Any) -> dict[str, Any]:
    """Extract common algorithm hyperparameters for logs."""
    keys = (
        "theta",
        "max_iterations",
        "solver_options",
        "alpha",
        "gamma",
        "epsilon",
        "lambda_",
        "episodes",
        "max_steps_per_episode",
    )
    return {
        key: getattr(algorithm, key)
        for key in keys
        if hasattr(algorithm, key)
    }


def _experiment_metadata(env: PlanningGridWorld | LearningGridWorld) -> dict[str, Any]:
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


def _serialize_states(states: set[State]) -> list[list[int]]:
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


def _add_learning_planning_baselines(
    learners: dict[str, Any],
    training_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Attach planning-baseline comparison metrics for learning algorithms."""
    planning_env = PlanningGridWorld(grid_size=(8, 8), seed=RANDOM_SEED)

    policy_evaluation = PolicyEvaluation(planning_env)
    pe_values = policy_evaluation.run()

    value_iteration = ValueIteration(planning_env)
    vi_values, vi_policy = value_iteration.run()
    control_states = [
        state for state in planning_env.get_states() if not planning_env.is_terminal(state)
    ]

    for name in ("td_zero", "td_lambda"):
        values = learners[name].get_value_function()
        training_metrics[name]["mse_vs_policy_evaluation"] = mean_squared_error(
            values,
            pe_values,
        )

    for name in ("sarsa", "q_learning"):
        values = learners[name].get_value_function()
        policy = learners[name].get_policy()
        training_metrics[name]["mse_vs_value_iteration"] = mean_squared_error(
            values,
            vi_values,
        )
        training_metrics[name]["policy_agreement_vs_value_iteration"] = (
            policy_agreement(policy, vi_policy, control_states)
        )

    return {
        "policy_evaluation": {
            "status": policy_evaluation.get_metrics()["status"],
            "iterations": policy_evaluation.get_metrics()["iterations"],
        },
        "value_iteration": {
            "status": value_iteration.get_metrics()["status"],
            "iterations": value_iteration.get_metrics()["iterations"],
        },
    }


def _save_planning_figures(
    env: PlanningGridWorld,
    planners: dict[str, Any],
    system_metrics: dict[str, Any],
) -> list[str]:
    """Save standard planning figures for the report."""
    figure_paths = [
        PLANNING_FIGURE_DIR / "policy_evaluation_values.png",
        PLANNING_FIGURE_DIR / "fixed_policy_arrows.png",
        PLANNING_FIGURE_DIR / "policy_iteration_values.png",
        PLANNING_FIGURE_DIR / "policy_iteration_policy.png",
        PLANNING_FIGURE_DIR / "value_iteration_values.png",
        PLANNING_FIGURE_DIR / "value_iteration_policy.png",
        PLANNING_FIGURE_DIR / "linear_programming_values.png",
        PLANNING_FIGURE_DIR / "linear_programming_policy.png",
        PLANNING_FIGURE_DIR / "value_iteration_residuals.png",
        PLANNING_FIGURE_DIR / "policy_iteration_policy_changes.png",
        PLANNING_FIGURE_DIR / "planning_runtime_comparison.png",
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
        planners["policy_iteration"].get_value_function(),
        env,
        "Policy Iteration: V*",
        figure_paths[2],
    )
    plot_policy_arrows(
        planners["policy_iteration"].get_policy(),
        env,
        "Policy Iteration Greedy Policy",
        figure_paths[3],
    )
    plot_value_heatmap(
        planners["value_iteration"].get_value_function(),
        env,
        "Value Iteration: V*",
        figure_paths[4],
    )
    plot_policy_arrows(
        planners["value_iteration"].get_policy(),
        env,
        "Value Iteration Greedy Policy",
        figure_paths[5],
    )
    plot_value_heatmap(
        planners["linear_programming"].get_value_function(),
        env,
        "Linear Programming: V*",
        figure_paths[6],
    )
    plot_policy_arrows(
        planners["linear_programming"].get_policy(),
        env,
        "Linear Programming Greedy Policy",
        figure_paths[7],
    )
    plot_bellman_residual(
        planners["value_iteration"].get_metrics()["bellman_residuals"],
        "Value Iteration Bellman Residual",
        figure_paths[8],
    )
    plot_bellman_residual(
        planners["policy_iteration"].get_metrics()["policy_changes_per_iteration"],
        "Policy Iteration Policy Changes",
        figure_paths[9],
    )
    plot_comparison_bar(
        {
            name: metrics["wall_time_sec"]
            for name, metrics in system_metrics.items()
        },
        "Planning Runtime Comparison",
        figure_paths[10],
        ylabel="Seconds",
    )
    return [str(path) for path in figure_paths]


def _save_learning_comparison_figures(
    training_metrics: dict[str, Any],
) -> list[str]:
    """Save within-learning comparison figures with valid pairings."""
    figure_paths = [
        LEARNING_FIGURE_DIR / "td_prediction_mse_vs_policy_evaluation.png",
        LEARNING_FIGURE_DIR / "td_final_error_comparison.png",
        LEARNING_FIGURE_DIR / "control_average_return_comparison.png",
        LEARNING_FIGURE_DIR / "control_success_rate_comparison.png",
        LEARNING_FIGURE_DIR / "control_trap_rate_comparison.png",
        LEARNING_FIGURE_DIR / "control_mse_vs_value_iteration.png",
        LEARNING_FIGURE_DIR / "control_policy_agreement_vs_value_iteration.png",
    ]

    plot_comparison_bar(
        {
            "td_zero": training_metrics["td_zero"]["mse_vs_policy_evaluation"],
            "td_lambda": training_metrics["td_lambda"]["mse_vs_policy_evaluation"],
        },
        "TD Prediction MSE vs Policy Evaluation",
        figure_paths[0],
        ylabel="MSE",
    )
    plot_comparison_bar(
        {
            "td_zero": training_metrics["td_zero"]["final_mean_absolute_td_error"],
            "td_lambda": training_metrics["td_lambda"]["final_mean_absolute_td_error"],
        },
        "TD Prediction Final Absolute TD Error",
        figure_paths[1],
        ylabel="Mean absolute TD error",
    )
    plot_comparison_bar(
        {
            "sarsa": training_metrics["sarsa"]["average_return"],
            "q_learning": training_metrics["q_learning"]["average_return"],
        },
        "SARSA vs Q-learning Average Return",
        figure_paths[2],
        ylabel="Average return",
    )
    plot_comparison_bar(
        {
            "sarsa": training_metrics["sarsa"]["success_rate"],
            "q_learning": training_metrics["q_learning"]["success_rate"],
        },
        "SARSA vs Q-learning Success Rate",
        figure_paths[3],
        ylabel="Rate",
    )
    plot_comparison_bar(
        {
            "sarsa": training_metrics["sarsa"]["trap_rate"],
            "q_learning": training_metrics["q_learning"]["trap_rate"],
        },
        "SARSA vs Q-learning Trap Rate",
        figure_paths[4],
        ylabel="Rate",
    )
    plot_comparison_bar(
        {
            "sarsa": training_metrics["sarsa"]["mse_vs_value_iteration"],
            "q_learning": training_metrics["q_learning"]["mse_vs_value_iteration"],
        },
        "Control MSE vs Value Iteration",
        figure_paths[5],
        ylabel="MSE",
    )
    plot_comparison_bar(
        {
            "sarsa": training_metrics["sarsa"]["policy_agreement_vs_value_iteration"],
            "q_learning": training_metrics["q_learning"][
                "policy_agreement_vs_value_iteration"
            ],
        },
        "Control Policy Agreement vs Value Iteration",
        figure_paths[6],
        ylabel="Agreement",
    )
    return [str(path) for path in figure_paths]


def _save_valid_comparison_figures() -> list[str]:
    """Save cross-folder comparison figures without invalid algorithm pairings."""
    planning_metrics = load_json(Path("logs") / "planning" / "training_metrics.json")
    planning_system = load_json(Path("logs") / "planning" / "system_metrics.json")
    learning_metrics = load_json(Path("logs") / "learning" / "training_metrics.json")
    learning_system = load_json(Path("logs") / "learning" / "system_metrics.json")

    figure_paths = [
        COMPARISON_FIGURE_DIR / "planning_optimal_runtime.png",
        COMPARISON_FIGURE_DIR / "planning_optimal_bellman_backups.png",
        COMPARISON_FIGURE_DIR / "planning_optimal_peak_memory.png",
        COMPARISON_FIGURE_DIR / "planning_optimal_value_error_vs_vi.png",
        COMPARISON_FIGURE_DIR / "planning_optimal_policy_agreement_vs_vi.png",
        COMPARISON_FIGURE_DIR / "td_prediction_mse_vs_policy_evaluation.png",
        COMPARISON_FIGURE_DIR / "td_prediction_runtime.png",
        COMPARISON_FIGURE_DIR / "control_learning_average_return.png",
        COMPARISON_FIGURE_DIR / "control_learning_success_rate.png",
        COMPARISON_FIGURE_DIR / "control_learning_trap_rate.png",
        COMPARISON_FIGURE_DIR / "control_learning_average_steps.png",
        COMPARISON_FIGURE_DIR / "control_learning_runtime.png",
        COMPARISON_FIGURE_DIR / "control_learning_mse_vs_value_iteration.png",
        COMPARISON_FIGURE_DIR / "control_learning_policy_agreement_vs_vi.png",
        COMPARISON_FIGURE_DIR / "control_learning_environment_steps.png",
    ]

    plot_comparison_bar(
        {
            name: planning_system[name]["wall_time_sec"]
            for name in ("policy_iteration", "value_iteration", "linear_programming")
        },
        "Planning Optimal-Control Runtime",
        figure_paths[0],
        ylabel="Seconds",
    )
    plot_comparison_bar(
        {
            name: planning_metrics[name].get("bellman_backups", 0)
            for name in ("policy_iteration", "value_iteration", "linear_programming")
        },
        "Planning Optimal-Control Bellman Backups",
        figure_paths[1],
        ylabel="Bellman backups",
    )
    plot_comparison_bar(
        {
            name: planning_system[name]["peak_memory_mb"]
            for name in ("policy_iteration", "value_iteration", "linear_programming")
        },
        "Planning Optimal-Control Peak Memory",
        figure_paths[2],
        ylabel="MB",
    )
    plot_comparison_bar(
        {
            "policy_iteration": planning_metrics["policy_iteration"][
                "value_error_vs_value_iteration"
            ],
            "linear_programming": planning_metrics["linear_programming"][
                "value_error_vs_value_iteration"
            ],
        },
        "Planning Value Error vs Value Iteration",
        figure_paths[3],
        ylabel="Max absolute error",
    )
    plot_comparison_bar(
        {
            "policy_iteration": planning_metrics["policy_iteration"][
                "policy_agreement_vs_value_iteration"
            ],
            "linear_programming": planning_metrics["linear_programming"][
                "policy_agreement_vs_value_iteration"
            ],
        },
        "Planning Policy Agreement vs Value Iteration",
        figure_paths[4],
        ylabel="Agreement",
    )
    plot_comparison_bar(
        {
            "td_zero": learning_metrics["td_zero"]["mse_vs_policy_evaluation"],
            "td_lambda": learning_metrics["td_lambda"]["mse_vs_policy_evaluation"],
        },
        "TD Prediction MSE vs Policy Evaluation",
        figure_paths[5],
        ylabel="MSE",
    )
    plot_comparison_bar(
        {
            "td_zero": learning_system["td_zero"]["wall_time_sec"],
            "td_lambda": learning_system["td_lambda"]["wall_time_sec"],
        },
        "TD Prediction Runtime",
        figure_paths[6],
        ylabel="Seconds",
    )
    plot_comparison_bar(
        {
            "sarsa": learning_metrics["sarsa"]["average_return"],
            "q_learning": learning_metrics["q_learning"]["average_return"],
        },
        "SARSA vs Q-learning Average Return",
        figure_paths[7],
        ylabel="Average return",
    )
    plot_comparison_bar(
        {
            "sarsa": learning_metrics["sarsa"]["success_rate"],
            "q_learning": learning_metrics["q_learning"]["success_rate"],
        },
        "SARSA vs Q-learning Success Rate",
        figure_paths[8],
        ylabel="Rate",
    )
    plot_comparison_bar(
        {
            "sarsa": learning_metrics["sarsa"]["trap_rate"],
            "q_learning": learning_metrics["q_learning"]["trap_rate"],
        },
        "SARSA vs Q-learning Trap Rate",
        figure_paths[9],
        ylabel="Rate",
    )
    plot_comparison_bar(
        {
            "sarsa": learning_metrics["sarsa"]["average_steps"],
            "q_learning": learning_metrics["q_learning"]["average_steps"],
        },
        "SARSA vs Q-learning Average Steps",
        figure_paths[10],
        ylabel="Steps",
    )
    plot_comparison_bar(
        {
            "sarsa": learning_system["sarsa"]["wall_time_sec"],
            "q_learning": learning_system["q_learning"]["wall_time_sec"],
        },
        "SARSA vs Q-learning Runtime",
        figure_paths[11],
        ylabel="Seconds",
    )
    plot_comparison_bar(
        {
            "sarsa": learning_metrics["sarsa"]["mse_vs_value_iteration"],
            "q_learning": learning_metrics["q_learning"]["mse_vs_value_iteration"],
        },
        "Control Learning MSE vs Value Iteration",
        figure_paths[12],
        ylabel="MSE",
    )
    plot_comparison_bar(
        {
            "sarsa": learning_metrics["sarsa"]["policy_agreement_vs_value_iteration"],
            "q_learning": learning_metrics["q_learning"][
                "policy_agreement_vs_value_iteration"
            ],
        },
        "Control Policy Agreement vs Value Iteration",
        figure_paths[13],
        ylabel="Agreement",
    )
    plot_comparison_bar(
        {
            "sarsa": learning_system["sarsa"]["environment_steps"],
            "q_learning": learning_system["q_learning"]["environment_steps"],
        },
        "Control Learning Environment Steps",
        figure_paths[14],
        ylabel="Environment steps",
    )
    return [str(path) for path in figure_paths]


def _save_learning_figures(name: str, learner: Any, metrics: dict[str, Any]) -> None:
    """Save standard learning figures for one trained learner."""
    env = learner.env
    if "episode_returns" in metrics:
        plot_learning_curve(
            metrics["episode_returns"],
            f"{name}: episode returns",
            LEARNING_FIGURE_DIR / f"{name}_episode_returns.png",
        )
    if "moving_average_returns" in metrics:
        plot_moving_average(
            metrics["moving_average_returns"],
            f"{name}: moving average returns",
            LEARNING_FIGURE_DIR / f"{name}_moving_average_returns.png",
        )
    if "episode_steps" in metrics:
        plot_episode_steps(
            metrics["episode_steps"],
            f"{name}: steps per episode",
            LEARNING_FIGURE_DIR / f"{name}_episode_steps.png",
        )
    if "td_errors" in metrics:
        plot_td_error_curve(
            metrics["td_errors"],
            f"{name}: TD errors",
            LEARNING_FIGURE_DIR / f"{name}_td_errors.png",
        )
    if "mean_absolute_td_error_per_episode" in metrics:
        plot_td_error_curve(
            metrics["mean_absolute_td_error_per_episode"],
            f"{name}: mean absolute TD error",
            LEARNING_FIGURE_DIR / f"{name}_mean_abs_td_error.png",
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
            LEARNING_FIGURE_DIR / f"{name}_outcome_rates.png",
        )
    if hasattr(learner, "get_value_function"):
        plot_value_heatmap(
            learner.get_value_function(),
            env,
            f"{name}: value function",
            LEARNING_FIGURE_DIR / f"{name}_value_heatmap.png",
        )
    policy = learner.get_policy()
    if policy:
        plot_policy_arrows(
            policy,
            env,
            f"{name}: learned policy",
            LEARNING_FIGURE_DIR / f"{name}_policy.png",
        )


def _best_by_metric(
    metrics_by_algorithm: dict[str, dict[str, Any]],
    metric: str,
) -> str | None:
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
