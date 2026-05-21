---
name: rl-gridworld-architecture
description: Use this skill when designing or modifying the OOP architecture for the 8x8 Grid-world Reinforcement Learning project, including envs/, agents/, utils/, logs/, notebooks/, and run_experiments.py.
---

# RL Grid-world Architecture Skill

## Purpose

Use this skill when the task involves repository structure, OOP design, environment interfaces, package boundaries, or integration between environment, agents, utilities, notebooks, logs, and reports.

## Project goal

Implement an 8x8 Grid-world Navigation project from scratch for a Machine Learning course report. The project compares two groups of classical Reinforcement Learning algorithms:

Planning Algorithms:
- Policy Evaluation
- Policy Iteration
- Value Iteration
- Linear Programming

Learning Algorithms:
- TD(0)
- SARSA
- Q-learning
- TD(lambda)

The environment should represent a finite Markov Decision Process: M = (S, A, P, r, gamma).

## Expected project structure

Use this target structure unless the user explicitly asks otherwise:

Reinforcement-Learning/
├── AGENTS.md
├── README.md
├── requirements.txt
├── .gitignore
├── run_experiments.py
├── envs/
│   ├── __init__.py
│   ├── grid_world.py
│   ├── planning_grid_world.py
│   └── learning_grid_world.py
├── agents/
│   ├── __init__.py
│   ├── planning.py
│   └── learning.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── profiling.py
│   ├── experiment_io.py
│   └── visualization.py
├── notebooks/
│   ├── planning.ipynb
│   └── learning.ipynb
├── logs/
│   ├── planning/
│   │   ├── training_metrics.json
│   │   ├── system_metrics.json
│   │   └── experiment_summary.json
│   └── learning/
│       ├── training_metrics.json
│       ├── system_metrics.json
│       └── experiment_summary.json
└── report/
    ├── main.tex
    ├── references.bib
    └── figures/

## OOP design rules

1. `envs/grid_world.py` should define the shared abstract/base interface.
2. `PlanningGridWorld` should expose full model access: `get_states()`, `get_actions(state)`, `get_transitions(state, action)`.
3. `LearningGridWorld` should expose sample-based interaction: `reset()`, `step(action)`, `is_terminal(state)`.
4. `agents/planning.py` should contain planning classes only.
5. `agents/learning.py` should contain learning classes only.
6. `utils/visualization.py` should not contain RL algorithm logic.
7. `utils/metrics.py` should compute MSE, MAE, max error, policy agreement, success rate, trap rate, average return, and Bellman residual.
8. `utils/profiling.py` should measure runtime, CPU time, current RAM, peak RAM, and update counts.
9. `utils/experiment_io.py` should handle saving/loading JSON logs and experiment summaries.

## Coding standards

- Use Python 3.10+.
- Use NumPy for matrix/table computation.
- Do not use Gymnasium, Stable-Baselines, RLlib, or scikit-learn for RL algorithms.
- Use SciPy only for Linear Programming if allowed by project constraints.
- Keep algorithms from scratch and readable.
- Add type hints and docstrings for public classes and methods.
- Use deterministic random seeds.
- Keep notebooks as presentation/analysis layers, not as the main implementation.

## Stop conditions

When creating or editing architecture, preserve separation of concerns:
- envs: environment dynamics
- agents: algorithms
- utils: metrics, logging, profiling, visualization
- notebooks: explanation and result presentation
- logs: generated outputs
