"""Experiment entry point for the 8x8 Grid-world RL project."""

from __future__ import annotations

import argparse
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
from utils.experiment_io import load_json, save_experiment_logs, save_json
from utils.logging_utils import (
    format_float,
    format_scientific,
    log_message,
    print_algorithm_start,
    print_metric_block,
    print_table,
)
from utils.metrics import max_abs_error, mean_squared_error, policy_agreement
from utils.profiling import profile_callable
from utils.visualization import (
    plot_bellman_residual,
    plot_comparison_bar,
    plot_episode_steps,
    plot_learning_curve,
    plot_moving_average,
    plot_policy_arrows,
    plot_sensitivity_bar_or_line,
    plot_success_trap_rates,
    plot_success_trap_curves,
    plot_td_mse_curve,
    plot_td_error_curve,
    plot_value_heatmap,
)


RANDOM_SEED = 42
LEARNING_EPISODES = 1_000
LEARNING_MAX_STEPS = 256
SENSITIVITY_EPISODES = 300
PLANNING_FIGURE_DIR = Path("report") / "figures" / "planning"
LEARNING_FIGURE_DIR = Path("report") / "figures" / "learning"
COMPARISON_FIGURE_DIR = Path("report") / "figures" / "comparison"


def run_planning_experiments(verbose: int = 0, log_interval: int = 10) -> dict[str, Any]:
    """Run full planning experiments on the default 8x8 Grid-world."""
    env = PlanningGridWorld(grid_size=(8, 8), seed=RANDOM_SEED)
    fixed_policy = _build_goal_directed_policy(env)
    algorithm_verbose = verbose if verbose >= 2 else 0
    planners = {
        "policy_evaluation": PolicyEvaluation(
            env,
            policy=fixed_policy,
            verbose=algorithm_verbose,
            log_interval=log_interval,
        ),
        "policy_iteration": PolicyIteration(
            env,
            verbose=algorithm_verbose,
            log_interval=log_interval,
        ),
        "value_iteration": ValueIteration(
            env,
            verbose=algorithm_verbose,
            log_interval=log_interval,
        ),
        "linear_programming": LinearProgrammingPlanner(
            env,
            verbose=algorithm_verbose,
            log_interval=log_interval,
        ),
    }

    training_metrics: dict[str, Any] = {}
    system_metrics: dict[str, Any] = {}
    for name, planner in planners.items():
        print_algorithm_start("Planning", _display_name(name), verbose)
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
        _print_planning_metric_block(name, training_metrics[name], system_metrics[name], verbose)

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
        "comparison_metrics": training_metrics["planning_comparison"],
        "figure_paths": figure_paths,
    }
    save_experiment_logs("planning", training_metrics, system_metrics, summary)
    return summary


def run_learning_experiments(
    verbose: int = 0,
    log_interval: int = 100,
    window_size: int = 100,
) -> dict[str, Any]:
    """Run model-free learning experiments on the default 8x8 Grid-world."""
    env = LearningGridWorld(grid_size=(8, 8), seed=RANDOM_SEED)
    prediction_baseline_env = PlanningGridWorld(grid_size=(8, 8), seed=RANDOM_SEED)
    prediction_baseline = PolicyEvaluation(prediction_baseline_env)
    prediction_baseline_values = prediction_baseline.run()
    algorithm_verbose = verbose if verbose >= 2 else 0
    learners = {
        "td_zero": TDZero(
            LearningGridWorld(grid_size=(8, 8), seed=RANDOM_SEED),
            episodes=LEARNING_EPISODES,
            max_steps_per_episode=LEARNING_MAX_STEPS,
            seed=RANDOM_SEED,
            verbose=algorithm_verbose,
            log_interval=log_interval,
            baseline_value_function=prediction_baseline_values,
        ),
        "td_lambda": TDLambda(
            LearningGridWorld(grid_size=(8, 8), seed=RANDOM_SEED),
            episodes=LEARNING_EPISODES,
            max_steps_per_episode=LEARNING_MAX_STEPS,
            seed=RANDOM_SEED,
            verbose=algorithm_verbose,
            log_interval=log_interval,
            baseline_value_function=prediction_baseline_values,
        ),
        "sarsa": SARSA(
            LearningGridWorld(grid_size=(8, 8), seed=RANDOM_SEED),
            episodes=LEARNING_EPISODES,
            max_steps_per_episode=LEARNING_MAX_STEPS,
            seed=RANDOM_SEED,
            verbose=algorithm_verbose,
            log_interval=log_interval,
            window_size=window_size,
        ),
        "q_learning": QLearning(
            LearningGridWorld(grid_size=(8, 8), seed=RANDOM_SEED),
            episodes=LEARNING_EPISODES,
            max_steps_per_episode=LEARNING_MAX_STEPS,
            seed=RANDOM_SEED,
            verbose=algorithm_verbose,
            log_interval=log_interval,
            window_size=window_size,
        ),
    }

    training_metrics: dict[str, Any] = {}
    system_metrics: dict[str, Any] = {}
    for name, learner in learners.items():
        print_algorithm_start("Learning", _display_name(name), verbose)
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
        _print_learning_metric_block(name, training_metrics[name], system_metrics[name], verbose)

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
        "best_control_by_training_avg_return": _best_by_metric(
            training_metrics,
            "training_avg_return",
        ),
        "lowest_td_prediction_error": _lowest_final_td_error(training_metrics),
        "planning_baselines": baseline_info,
        "figure_dir": str(LEARNING_FIGURE_DIR),
        "comparison_figure_paths": learning_comparison_paths,
    }
    save_experiment_logs("learning", training_metrics, system_metrics, summary)
    return summary


def run_comparison_experiments(verbose: int = 0) -> dict[str, Any]:
    """Prepare report figures for valid cross-group comparisons."""
    log_message("[Comparison] Generating valid comparison figures...", verbose)
    figure_paths = _save_valid_comparison_figures()
    log_message(
        f"[Comparison] Done: figures={len(figure_paths)}",
        verbose,
    )
    return {
        "status": "completed",
        "description": (
            "Planning algorithms use the full model. Learning algorithms use "
            "sampled environment interactions and are compared only against "
            "matching planning baselines."
        ),
        "figure_paths": figure_paths,
    }


def run_td_lambda_sweep(
    verbose: int = 0,
    log_interval: int = 100,
    window_size: int = 100,
    lambda_values: list[float] | None = None,
) -> dict[str, Any]:
    """Run an optional lightweight TD(lambda) lambda sensitivity sweep."""
    _ = window_size
    lambda_values = lambda_values or [0.0, 0.3, 0.5, 0.7, 0.9]
    planning_env = PlanningGridWorld(grid_size=(8, 8), seed=RANDOM_SEED)
    baseline_values = PolicyEvaluation(planning_env).run()
    algorithm_verbose = verbose if verbose >= 2 else 0
    rows: list[dict[str, Any]] = []

    for lambda_value in lambda_values:
        learner = TDLambda(
            LearningGridWorld(grid_size=(8, 8), seed=RANDOM_SEED),
            lambda_=lambda_value,
            episodes=SENSITIVITY_EPISODES,
            max_steps_per_episode=LEARNING_MAX_STEPS,
            seed=RANDOM_SEED,
            verbose=algorithm_verbose,
            log_interval=log_interval,
            baseline_value_function=baseline_values,
        )
        learner.train()
        metrics = learner.get_metrics()
        rows.append(
            {
                "lambda": lambda_value,
                "final_mse_vs_pe": metrics["mse_vs_policy_evaluation"],
                "final_td_error": metrics["final_mean_absolute_td_error"],
                "runtime": metrics["runtime_sec"],
                "environment_steps": metrics["environment_steps"],
            }
        )

    figure_path = LEARNING_FIGURE_DIR / "td_lambda_sensitivity_mse.png"
    plot_sensitivity_bar_or_line(
        [row["lambda"] for row in rows],
        [row["final_mse_vs_pe"] for row in rows],
        "TD(lambda) Sensitivity: MSE vs Policy Evaluation",
        figure_path,
        xlabel="lambda",
        ylabel="Final MSE vs PE",
    )
    summary = {
        "status": "completed",
        "episodes_per_run": SENSITIVITY_EPISODES,
        "lambda_values": lambda_values,
        "metrics": rows,
        "figure_paths": [str(figure_path)],
    }
    best_row = _best_row(rows, "final_mse_vs_pe", minimize=True)
    if best_row is not None:
        summary.update(
            {
                "best_lambda_by_mse": best_row["lambda"],
                "best_mse_vs_pe": best_row["final_mse_vs_pe"],
                "best_final_td_error": best_row["final_td_error"],
                "best_runtime": best_row["runtime"],
            }
        )
    save_json(summary, Path("logs") / "learning" / "td_lambda_sweep.json")
    print_metric_block(
        "[Sensitivity] TD(lambda) sweep",
        (
            ("runs", len(rows)),
            ("episodes_per_run", SENSITIVITY_EPISODES),
            ("best_lambda", _format_optional_float(summary.get("best_lambda_by_mse"))),
            ("best_mse_vs_pe", format_scientific(summary.get("best_mse_vs_pe"))),
            ("best_td_error", format_scientific(summary.get("best_final_td_error"))),
        ),
        verbose,
    )
    return summary


def run_control_sensitivity(
    verbose: int = 0,
    log_interval: int = 100,
    window_size: int = 100,
    epsilon_values: list[float] | None = None,
) -> dict[str, Any]:
    """Run an optional lightweight SARSA/Q-learning epsilon sensitivity sweep."""
    epsilon_values = epsilon_values or [0.01, 0.05, 0.1, 0.2]
    planning_env = PlanningGridWorld(grid_size=(8, 8), seed=RANDOM_SEED)
    vi_values, vi_policy = ValueIteration(planning_env).run()
    control_states = [
        state for state in planning_env.get_states() if not planning_env.is_terminal(state)
    ]
    algorithm_verbose = verbose if verbose >= 2 else 0
    rows: list[dict[str, Any]] = []

    for algorithm_name, learner_class in (("sarsa", SARSA), ("q_learning", QLearning)):
        for epsilon in epsilon_values:
            learner = learner_class(
                LearningGridWorld(grid_size=(8, 8), seed=RANDOM_SEED),
                epsilon=epsilon,
                episodes=SENSITIVITY_EPISODES,
                max_steps_per_episode=LEARNING_MAX_STEPS,
                seed=RANDOM_SEED,
                verbose=algorithm_verbose,
                log_interval=log_interval,
                window_size=window_size,
            )
            learner.train()
            values = learner.get_value_function()
            policy = learner.get_policy()
            metrics = learner.get_metrics()
            rows.append(
                {
                    "algorithm": algorithm_name,
                    "epsilon": epsilon,
                    "final_window_avg_return": metrics["final_window_avg_return"],
                    "final_window_success_rate": metrics["final_window_success_rate"],
                    "final_window_trap_rate": metrics["final_window_trap_rate"],
                    "policy_agreement_vs_value_iteration": policy_agreement(
                        policy,
                        vi_policy,
                        control_states,
                    ),
                    "mse_vs_value_iteration": mean_squared_error(values, vi_values),
                    "runtime": metrics["runtime_sec"],
                    "environment_steps": metrics["environment_steps"],
                }
            )

    figure_paths: list[str] = []
    for algorithm_name in ("sarsa", "q_learning"):
        algorithm_rows = [row for row in rows if row["algorithm"] == algorithm_name]
        figure_path = LEARNING_FIGURE_DIR / f"{algorithm_name}_epsilon_sensitivity_return.png"
        plot_sensitivity_bar_or_line(
            [row["epsilon"] for row in algorithm_rows],
            [row["final_window_avg_return"] for row in algorithm_rows],
            f"{_display_name(algorithm_name)} Epsilon Sensitivity: Final Window Return",
            figure_path,
            xlabel="epsilon",
            ylabel="Final-window return",
        )
        figure_paths.append(str(figure_path))

    summary = {
        "status": "completed",
        "episodes_per_run": SENSITIVITY_EPISODES,
        "epsilon_values": epsilon_values,
        "metrics": rows,
        "figure_paths": figure_paths,
    }
    for algorithm_name in ("sarsa", "q_learning"):
        algorithm_rows = [row for row in rows if row["algorithm"] == algorithm_name]
        best_row = _best_row(algorithm_rows, "final_window_avg_return")
        if best_row is not None:
            summary[f"{algorithm_name}_best_epsilon_by_window_return"] = best_row[
                "epsilon"
            ]
            summary[f"{algorithm_name}_best_window_return"] = best_row[
                "final_window_avg_return"
            ]
            summary[f"{algorithm_name}_best_window_success"] = best_row[
                "final_window_success_rate"
            ]
            summary[f"{algorithm_name}_best_window_trap"] = best_row[
                "final_window_trap_rate"
            ]
            summary[f"{algorithm_name}_best_policy_agreement_vs_vi"] = best_row[
                "policy_agreement_vs_value_iteration"
            ]
    save_json(summary, Path("logs") / "learning" / "control_sensitivity.json")
    print_metric_block(
        "[Sensitivity] Control epsilon sweep",
        (
            ("runs", len(rows)),
            ("episodes_per_run", SENSITIVITY_EPISODES),
            (
                "SARSA best_epsilon",
                _format_optional_float(
                    summary.get("sarsa_best_epsilon_by_window_return")
                ),
            ),
            ("SARSA best_return", format_float(summary.get("sarsa_best_window_return"))),
            (
                "QLearning best_epsilon",
                _format_optional_float(
                    summary.get("q_learning_best_epsilon_by_window_return")
                ),
            ),
            (
                "QLearning best_return",
                format_float(summary.get("q_learning_best_window_return")),
            ),
        ),
        verbose,
    )
    return summary


def run_gamma_sensitivity(
    verbose: int = 0,
    log_interval: int = 10,
    gamma_values: list[float] | None = None,
) -> dict[str, Any]:
    """Run an optional lightweight Value Iteration gamma sensitivity sweep."""
    gamma_values = gamma_values or [0.5, 0.7, 0.9, 0.99]
    algorithm_verbose = verbose if verbose >= 2 else 0
    rows: list[dict[str, Any]] = []

    for gamma in gamma_values:
        planner = ValueIteration(
            PlanningGridWorld(grid_size=(8, 8), seed=RANDOM_SEED, gamma=gamma),
            verbose=algorithm_verbose,
            log_interval=log_interval,
        )
        profile = profile_callable(planner.run)
        metrics = planner.get_metrics()
        rows.append(
            {
                "gamma": gamma,
                "iterations": metrics["iterations"],
                "final_residual": metrics["final_bellman_residual"],
                "bellman_backups": metrics["bellman_backups"],
                "runtime": profile.wall_time_sec,
            }
        )

    figure_path = PLANNING_FIGURE_DIR / "value_iteration_gamma_sensitivity_iterations.png"
    plot_sensitivity_bar_or_line(
        [row["gamma"] for row in rows],
        [row["iterations"] for row in rows],
        "Value Iteration Gamma Sensitivity",
        figure_path,
        xlabel="gamma",
        ylabel="Iterations",
    )
    summary = {
        "status": "completed",
        "gamma_values": gamma_values,
        "metrics": rows,
        "figure_paths": [str(figure_path)],
    }
    fastest_row = _best_row(rows, "iterations", minimize=True)
    slowest_row = _best_row(rows, "iterations")
    if fastest_row is not None:
        summary.update(
            {
                "fastest_gamma": fastest_row["gamma"],
                "min_iterations": fastest_row["iterations"],
                "fastest_runtime": fastest_row["runtime"],
            }
        )
    if slowest_row is not None:
        summary.update(
            {
                "slowest_gamma": slowest_row["gamma"],
                "max_iterations": slowest_row["iterations"],
                "slowest_runtime": slowest_row["runtime"],
            }
        )
    save_json(summary, Path("logs") / "planning" / "gamma_sensitivity.json")
    print_metric_block(
        "[Sensitivity] Value Iteration gamma sweep",
        (
            ("runs", len(rows)),
            ("fastest_gamma", _format_optional_float(summary.get("fastest_gamma"))),
            ("slowest_gamma", _format_optional_float(summary.get("slowest_gamma"))),
            ("min_iterations", summary.get("min_iterations", "n/a")),
            ("max_iterations", summary.get("max_iterations", "n/a")),
        ),
        verbose,
    )
    return summary


def run_multiseed_smoke(
    verbose: int = 0,
    window_size: int = 100,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Run a lightweight multi-seed stability smoke test for control learners."""
    seeds = seeds or [0, 1, 2]
    planning_env = PlanningGridWorld(grid_size=(8, 8), seed=RANDOM_SEED)
    vi_values, vi_policy = ValueIteration(planning_env).run()
    control_states = [
        state for state in planning_env.get_states() if not planning_env.is_terminal(state)
    ]
    rows: list[dict[str, Any]] = []

    for algorithm_name, learner_class in (("sarsa", SARSA), ("q_learning", QLearning)):
        for seed in seeds:
            learner = learner_class(
                LearningGridWorld(grid_size=(8, 8), seed=seed),
                episodes=SENSITIVITY_EPISODES,
                max_steps_per_episode=LEARNING_MAX_STEPS,
                seed=seed,
                window_size=window_size,
            )
            learner.train()
            values = learner.get_value_function()
            policy = learner.get_policy()
            metrics = learner.get_metrics()
            rows.append(
                {
                    "algorithm": algorithm_name,
                    "seed": seed,
                    "final_window_avg_return": metrics["final_window_avg_return"],
                    "final_window_success_rate": metrics["final_window_success_rate"],
                    "final_window_trap_rate": metrics["final_window_trap_rate"],
                    "policy_agreement_vs_value_iteration": policy_agreement(
                        policy,
                        vi_policy,
                        control_states,
                    ),
                    "mse_vs_value_iteration": mean_squared_error(values, vi_values),
                    "runtime": metrics["runtime_sec"],
                    "environment_steps": metrics["environment_steps"],
                }
            )

    aggregate = {
        algorithm_name: _multiseed_aggregate(
            [row for row in rows if row["algorithm"] == algorithm_name]
        )
        for algorithm_name in ("sarsa", "q_learning")
    }
    summary = {
        "status": "completed",
        "seeds": seeds,
        "episodes_per_run": SENSITIVITY_EPISODES,
        "metrics": rows,
        "aggregate": aggregate,
    }
    save_json(summary, Path("logs") / "learning" / "multiseed_smoke.json")
    print_metric_block(
        "[Stability] Multi-seed smoke",
        (
            ("seeds", len(seeds)),
            ("episodes/run", SENSITIVITY_EPISODES),
            ("SARSA mean_return", format_float(aggregate["sarsa"]["mean_window_return"])),
            ("SARSA std_return", format_float(aggregate["sarsa"]["std_window_return"])),
            (
                "QLearning mean_return",
                format_float(aggregate["q_learning"]["mean_window_return"]),
            ),
            (
                "QLearning std_return",
                format_float(aggregate["q_learning"]["std_window_return"]),
            ),
        ),
        verbose,
    )
    return summary


def main() -> None:
    """Run all experiment groups."""
    args = _parse_args()
    planning_summary = run_planning_experiments(
        verbose=args.verbose,
        log_interval=args.log_interval,
    )
    learning_summary = run_learning_experiments(
        verbose=args.verbose,
        log_interval=args.log_interval,
        window_size=args.window_size,
    )
    comparison_summary = run_comparison_experiments(verbose=args.verbose)
    if args.run_td_lambda_sweep:
        run_td_lambda_sweep(
            verbose=args.verbose,
            log_interval=args.log_interval,
            window_size=args.window_size,
        )
    if args.run_control_sensitivity:
        run_control_sensitivity(
            verbose=args.verbose,
            log_interval=args.log_interval,
            window_size=args.window_size,
        )
    if args.run_gamma_sensitivity:
        run_gamma_sensitivity(verbose=args.verbose, log_interval=args.log_interval)
    if args.run_multiseed_smoke:
        run_multiseed_smoke(verbose=args.verbose, window_size=args.window_size)
    _print_final_summary(planning_summary, learning_summary, comparison_summary, args.verbose)


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for experiment progress logging."""
    parser = argparse.ArgumentParser(description="Run Grid-world RL experiments.")
    parser.add_argument(
        "--verbose",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="0=silent, 1=start/done summaries, 2=periodic progress logs.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Iterations/episodes between progress logs when --verbose 2.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=100,
        help="Episode window for moving averages and final-window learning metrics.",
    )
    parser.add_argument(
        "--run-td-lambda-sweep",
        action="store_true",
        help="Run optional lightweight TD(lambda) lambda sensitivity.",
    )
    parser.add_argument(
        "--run-control-sensitivity",
        action="store_true",
        help="Run optional lightweight SARSA/Q-learning epsilon sensitivity.",
    )
    parser.add_argument(
        "--run-gamma-sensitivity",
        action="store_true",
        help="Run optional lightweight Value Iteration gamma sensitivity.",
    )
    parser.add_argument(
        "--run-multiseed-smoke",
        action="store_true",
        help="Run optional lightweight SARSA/Q-learning multi-seed smoke test.",
    )
    args = parser.parse_args()
    if args.log_interval <= 0:
        parser.error("--log-interval must be a positive integer")
    if args.window_size <= 0:
        parser.error("--window-size must be a positive integer")
    return args


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
        "verbose",
        "log_interval",
        "window_size",
        "epsilon_decay",
        "epsilon_min",
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


def _display_name(name: str) -> str:
    """Return a compact class-like display name for terminal progress."""
    return {
        "policy_evaluation": "PolicyEvaluation",
        "policy_iteration": "PolicyIteration",
        "value_iteration": "ValueIteration",
        "linear_programming": "LinearProgrammingPlanner",
        "td_zero": "TDZero",
        "td_lambda": "TDLambda",
        "sarsa": "SARSA",
        "q_learning": "QLearning",
    }.get(name, name)


def _print_planning_metric_block(
    name: str,
    training_metrics: dict[str, Any],
    system_metrics: dict[str, Any],
    verbose: int,
) -> None:
    """Print one planning completion block."""
    display_name = _display_name(name)
    runtime = system_metrics["wall_time_sec"]
    status = training_metrics.get("status", "completed")
    backups = training_metrics.get("bellman_backups", 0)

    if name == "policy_iteration":
        iterations = training_metrics.get("policy_improvement_iterations", 0)
        residual = training_metrics.get("final_policy_evaluation_residual")
    elif name == "linear_programming":
        print_metric_block(
            f"[Planning] {display_name}",
            (
                ("status", status),
                ("variables", training_metrics.get("number_of_variables", 0)),
                ("constraints", training_metrics.get("number_of_constraints", 0)),
                ("runtime", f"{runtime:.3f}s"),
            ),
            verbose,
        )
        return
    else:
        iterations = training_metrics.get("iterations", 0)
        residual = training_metrics.get("final_bellman_residual")

    print_metric_block(
        f"[Planning] {display_name}",
        (
            ("status", status),
            ("iterations", iterations),
            ("residual", format_scientific(residual)),
            ("backups", backups),
            ("runtime", f"{runtime:.3f}s"),
        ),
        verbose,
    )


def _print_learning_metric_block(
    name: str,
    training_metrics: dict[str, Any],
    system_metrics: dict[str, Any],
    verbose: int,
) -> None:
    """Print one learning completion block."""
    display_name = _display_name(name)
    runtime = system_metrics["wall_time_sec"]
    episodes = training_metrics.get("episodes", 0)
    environment_steps = training_metrics.get("environment_steps", 0)

    if name in {"sarsa", "q_learning"}:
        print_metric_block(
            f"[Learning] {display_name}",
            (
                ("episodes", episodes),
                ("env_steps", environment_steps),
                ("training_avg_return", format_float(training_metrics.get("training_avg_return"))),
                ("final_window_return", format_float(training_metrics.get("final_window_avg_return"))),
                ("training_success_rate", format_float(training_metrics.get("training_success_rate"))),
                ("final_window_success", format_float(training_metrics.get("final_window_success_rate"))),
                ("training_trap_rate", format_float(training_metrics.get("training_trap_rate"))),
                ("final_window_trap", format_float(training_metrics.get("final_window_trap_rate"))),
                ("runtime", f"{runtime:.3f}s"),
            ),
            verbose,
        )
        return

    print_metric_block(
        f"[Learning] {display_name}",
        (
            ("episodes", episodes),
            ("env_steps", environment_steps),
            ("td_error", format_scientific(training_metrics.get("final_mean_absolute_td_error"))),
            ("mse_vs_pe", format_scientific(training_metrics.get("mse_vs_policy_evaluation"))),
            ("runtime", f"{runtime:.3f}s"),
        ),
        verbose,
    )


def _print_final_summary(
    planning_summary: dict[str, Any],
    learning_summary: dict[str, Any],
    comparison_summary: dict[str, Any],
    verbose: int,
) -> None:
    """Print final experiment status and compact summary tables."""
    if verbose < 1:
        return

    print_metric_block(
        "[Experiments] Completed successfully.",
        (
            ("planning", planning_summary["status"]),
            ("learning", learning_summary["status"]),
            ("comparison", comparison_summary["status"]),
            ("figures", len(comparison_summary.get("figure_paths", []))),
        ),
        verbose,
    )

    planning_metrics = load_json(Path("logs") / "planning" / "training_metrics.json")
    planning_system = load_json(Path("logs") / "planning" / "system_metrics.json")
    learning_metrics = load_json(Path("logs") / "learning" / "training_metrics.json")
    learning_system = load_json(Path("logs") / "learning" / "system_metrics.json")

    print_table(
        "Planning summary",
        ("algorithm", "status", "iterations", "solver_iter", "residual", "runtime"),
        _planning_summary_rows(planning_metrics, planning_system),
        (24, 12, 10, 11, 10, 8),
        verbose,
    )
    print_table(
        "Learning summary",
        (
            "algorithm",
            "train_return",
            "window_return",
            "window_success",
            "window_trap",
            "mse_vs_pe",
            "mse_vs_vi",
            "policy_agree",
            "env_steps",
            "runtime",
        ),
        _learning_summary_rows(learning_metrics, learning_system),
        (10, 12, 13, 14, 11, 10, 10, 12, 9, 8),
        verbose,
    )


def _planning_summary_rows(
    planning_metrics: dict[str, dict[str, Any]],
    planning_system: dict[str, dict[str, Any]],
) -> list[tuple[Any, ...]]:
    """Build final planning summary rows."""
    rows: list[tuple[Any, ...]] = []
    for name in ("policy_evaluation", "policy_iteration", "value_iteration", "linear_programming"):
        metrics = planning_metrics[name]
        system = planning_system[name]
        if name == "policy_iteration":
            iterations = metrics.get("policy_improvement_iterations", "n/a")
            solver_iter = "n/a"
            residual = metrics.get("final_policy_evaluation_residual")
        elif name == "linear_programming":
            iterations = "n/a"
            solver_iter = metrics.get("solver_iterations", "n/a")
            residual = None
        else:
            iterations = metrics.get("iterations", "n/a")
            solver_iter = "n/a"
            residual = metrics.get("final_bellman_residual")

        rows.append(
            (
                _display_name(name),
                metrics.get("status", "n/a"),
                iterations,
                solver_iter,
                format_scientific(residual),
                f"{system.get('wall_time_sec', 0.0):.3f}s",
            )
        )
    return rows


def _learning_summary_rows(
    learning_metrics: dict[str, dict[str, Any]],
    learning_system: dict[str, dict[str, Any]],
) -> list[tuple[Any, ...]]:
    """Build final learning summary rows."""
    rows: list[tuple[Any, ...]] = []
    for name in ("td_zero", "td_lambda", "sarsa", "q_learning"):
        metrics = learning_metrics[name]
        system = learning_system[name]
        rows.append(
            (
                _display_name(name),
                format_float(metrics.get("training_avg_return")),
                format_float(metrics.get("final_window_avg_return")),
                format_float(metrics.get("final_window_success_rate")),
                format_float(metrics.get("final_window_trap_rate")),
                format_scientific(metrics.get("mse_vs_policy_evaluation")),
                format_scientific(metrics.get("mse_vs_value_iteration")),
                format_float(metrics.get("policy_agreement_vs_value_iteration")),
                system.get("environment_steps", metrics.get("environment_steps", 0)),
                f"{system.get('wall_time_sec', 0.0):.3f}s",
            )
        )
    return rows


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
        training_metrics[name]["value_mse_vs_value_iteration"] = mean_squared_error(
            values,
            vi_values,
        )
        training_metrics[name]["policy_agreement_vs_value_iteration"] = (
            policy_agreement(policy, vi_policy, comparison_states)
        )

    value_error_pi_vs_vi = training_metrics["policy_iteration"][
        "value_error_vs_value_iteration"
    ]
    value_error_lp_vs_vi = training_metrics["linear_programming"][
        "value_error_vs_value_iteration"
    ]
    value_mse_pi_vs_vi = training_metrics["policy_iteration"][
        "value_mse_vs_value_iteration"
    ]
    value_mse_lp_vs_vi = training_metrics["linear_programming"][
        "value_mse_vs_value_iteration"
    ]
    policy_agreement_pi_vs_vi = training_metrics["policy_iteration"][
        "policy_agreement_vs_value_iteration"
    ]
    policy_agreement_lp_vs_vi = training_metrics["linear_programming"][
        "policy_agreement_vs_value_iteration"
    ]
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
        "value_error_pi_vs_vi_inf_norm": value_error_pi_vs_vi,
        "value_error_lp_vs_vi_inf_norm": value_error_lp_vs_vi,
        "value_mse_pi_vs_vi": value_mse_pi_vs_vi,
        "value_mse_lp_vs_vi": value_mse_lp_vs_vi,
        "policy_agreement_pi_vs_vi": policy_agreement_pi_vs_vi,
        "policy_agreement_lp_vs_vi": policy_agreement_lp_vs_vi,
        "policy_iteration_vs_value_iteration_max_abs_error": value_error_pi_vs_vi,
        "linear_programming_vs_value_iteration_max_abs_error": value_error_lp_vs_vi,
        "policy_iteration_vs_value_iteration_policy_agreement": policy_agreement_pi_vs_vi,
        "linear_programming_vs_value_iteration_policy_agreement": policy_agreement_lp_vs_vi,
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
        LEARNING_FIGURE_DIR / "control_training_avg_return_comparison.png",
        LEARNING_FIGURE_DIR / "control_training_success_rate_comparison.png",
        LEARNING_FIGURE_DIR / "control_training_trap_rate_comparison.png",
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
            "sarsa": training_metrics["sarsa"]["training_avg_return"],
            "q_learning": training_metrics["q_learning"]["training_avg_return"],
        },
        "SARSA vs Q-learning Training Average Return",
        figure_paths[2],
        ylabel="Training average return",
    )
    plot_comparison_bar(
        {
            "sarsa": training_metrics["sarsa"]["training_success_rate"],
            "q_learning": training_metrics["q_learning"]["training_success_rate"],
        },
        "SARSA vs Q-learning Training Success Rate",
        figure_paths[3],
        ylabel="Rate",
    )
    plot_comparison_bar(
        {
            "sarsa": training_metrics["sarsa"]["training_trap_rate"],
            "q_learning": training_metrics["q_learning"]["training_trap_rate"],
        },
        "SARSA vs Q-learning Training Trap Rate",
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
        COMPARISON_FIGURE_DIR / "control_learning_training_avg_return.png",
        COMPARISON_FIGURE_DIR / "control_learning_training_success_rate.png",
        COMPARISON_FIGURE_DIR / "control_learning_training_trap_rate.png",
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
            "sarsa": learning_metrics["sarsa"]["training_avg_return"],
            "q_learning": learning_metrics["q_learning"]["training_avg_return"],
        },
        "SARSA vs Q-learning Training Average Return",
        figure_paths[7],
        ylabel="Training average return",
    )
    plot_comparison_bar(
        {
            "sarsa": learning_metrics["sarsa"]["training_success_rate"],
            "q_learning": learning_metrics["q_learning"]["training_success_rate"],
        },
        "SARSA vs Q-learning Training Success Rate",
        figure_paths[8],
        ylabel="Rate",
    )
    plot_comparison_bar(
        {
            "sarsa": learning_metrics["sarsa"]["training_trap_rate"],
            "q_learning": learning_metrics["q_learning"]["training_trap_rate"],
        },
        "SARSA vs Q-learning Training Trap Rate",
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
    if (
        metrics.get("mse_checkpoint_episodes")
        and metrics.get("mse_vs_policy_evaluation_checkpoints")
    ):
        plot_td_mse_curve(
            metrics["mse_checkpoint_episodes"],
            metrics["mse_vs_policy_evaluation_checkpoints"],
            f"{name}: MSE vs Policy Evaluation",
            LEARNING_FIGURE_DIR / f"{name}_mse_vs_policy_evaluation.png",
        )
    if (
        metrics.get("window_metric_episodes")
        and metrics.get("window_success_rates")
        and metrics.get("window_trap_rates")
    ):
        plot_success_trap_curves(
            metrics["window_metric_episodes"],
            metrics["window_success_rates"],
            metrics["window_trap_rates"],
            f"{name}: trailing-window outcome rates",
            LEARNING_FIGURE_DIR / f"{name}_window_outcome_rates.png",
        )
    if "training_success_rate" in metrics and "training_trap_rate" in metrics:
        timeout_rate = max(
            0.0,
            1.0 - metrics["training_success_rate"] - metrics["training_trap_rate"],
        )
        plot_success_trap_rates(
            {
                "success": metrics["training_success_rate"],
                "trap": metrics["training_trap_rate"],
                "timeout": timeout_rate,
            },
            f"{name}: training outcome rates",
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


def _best_row(
    rows: list[dict[str, Any]],
    metric: str,
    minimize: bool = False,
) -> dict[str, Any] | None:
    """Return the row with the best scalar metric."""
    candidates = [
        row
        for row in rows
        if isinstance(row.get(metric), (int, float))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row[metric]) if minimize else max(
        candidates,
        key=lambda row: row[metric],
    )


def _format_optional_float(value: Any) -> str:
    """Format an optional scalar float for concise terminal summaries."""
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _multiseed_aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate control-learning metrics across smoke-test seeds."""
    return {
        "mean_window_return": _mean_metric(rows, "final_window_avg_return"),
        "std_window_return": _std_metric(rows, "final_window_avg_return"),
        "mean_window_success": _mean_metric(rows, "final_window_success_rate"),
        "std_window_success": _std_metric(rows, "final_window_success_rate"),
        "mean_window_trap": _mean_metric(rows, "final_window_trap_rate"),
        "std_window_trap": _std_metric(rows, "final_window_trap_rate"),
        "mean_policy_agreement_vs_vi": _mean_metric(
            rows,
            "policy_agreement_vs_value_iteration",
        ),
        "mean_mse_vs_vi": _mean_metric(rows, "mse_vs_value_iteration"),
    }


def _mean_metric(rows: list[dict[str, Any]], metric: str) -> float:
    """Return the mean of one scalar metric across rows."""
    values = [row[metric] for row in rows if isinstance(row.get(metric), (int, float))]
    return float(np.mean(values)) if values else 0.0


def _std_metric(rows: list[dict[str, Any]], metric: str) -> float:
    """Return the population standard deviation of one scalar metric across rows."""
    values = [row[metric] for row in rows if isinstance(row.get(metric), (int, float))]
    return float(np.std(values)) if values else 0.0


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
