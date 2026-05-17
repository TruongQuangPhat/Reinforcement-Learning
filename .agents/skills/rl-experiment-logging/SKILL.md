---
name: rl-experiment-logging
description: Use this skill when designing or modifying experiment execution, metrics collection, logging, profiling, reproducibility, JSON outputs, or run_experiments.py for the Grid-world RL project.
---

# RL Experiment Logging and Profiling Skill

## Scope

Use this skill for:
- run_experiments.py
- JSON logging
- metrics aggregation
- reproducibility
- profiling runtime and memory
- experiment summaries

## Required log structure

logs/
├── planning/
│   ├── training_metrics.json
│   ├── system_metrics.json
│   └── experiment_summary.json
└── learning/
    ├── training_metrics.json
    ├── system_metrics.json
    └── experiment_summary.json

## Reproducibility requirements

Every experiment must record:
- random_seed
- grid_size
- start_state
- goal_states
- trap_states
- wall_states
- reward_config
- gamma
- algorithm_name
- hyperparameters
- timestamp
- Python version
- NumPy version
- platform if available

## System metrics

Measure:
- wall_time_sec using time.perf_counter()
- cpu_time_sec using time.process_time()
- current_memory_mb and peak_memory_mb using tracemalloc
- optionally process RSS memory using psutil if installed
- Bellman backups for planning
- environment steps for learning
- TD/Q updates for learning
- LP variables and constraints for Linear Programming

## Training metrics

Planning:
- bellman_residuals
- iterations
- policy_changes_per_iteration
- value_function
- policy
- value_errors_vs_baseline
- policy_agreement

Learning:
- episode_returns
- moving_average_returns
- episode_steps
- success_rate
- trap_rate
- td_errors
- mse_vs_planning_baseline
- policy_agreement_vs_value_iteration

## Experiment summary

Summarize:
- best planning algorithm by runtime and agreement
- best learning algorithm by average return
- most stable learning algorithm by variance across seeds
- TD(lambda) vs TD(0) convergence
- Q-learning vs SARSA behavior
- limitations and interpretation

## Logging rules

- Do not overwrite logs silently unless explicitly requested.
- Prefer timestamped output folders when running multiple experiment batches.
- Keep raw metrics separate from summary metrics.
- Use JSON-serializable structures only.
- Convert tuple states to strings when saving JSON, e.g. "(3, 4)".
