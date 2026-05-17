# AGENTS.md

## Project identity

This repository is a from-scratch Reinforcement Learning project for an 8x8 Grid-world Navigation experiment. The project supports two algorithm groups:

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

The project is for an academic Machine Learning report, so correctness, clarity, reproducibility, and clear mathematical alignment matter more than over-engineering.

## Repository architecture

Target structure:

Reinforcement-Learning/
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
│   └── learning/
└── report/
    ├── main.tex
    ├── references.bib
    └── figures/

## Coding rules

- Use Python 3.10+.
- Use NumPy for array/table computation.
- Implement RL algorithms from scratch.
- Do not use Gymnasium, Stable-Baselines, RLlib, or scikit-learn for RL algorithms.
- SciPy is allowed only for Linear Programming if required.
- Use OOP with clear interfaces.
- Keep notebooks for explanation and visualization, not core implementation.
- Add type hints and docstrings.
- Use deterministic random seeds.
- Save experiment results as JSON under logs/planning/ and logs/learning/.
- Save figures under report/figures/.

## Environment rules

- envs/grid_world.py defines shared abstractions/interfaces.
- envs/planning_grid_world.py exposes full transition model through transition_prob().
- envs/learning_grid_world.py exposes sample interaction through reset() and step().
- Planning algorithms may use transition_prob().
- Learning algorithms must not use transition_prob() during training.

## Experiment rules

Planning metrics:
- Bellman residuals
- iterations
- policy changes
- Bellman backups
- value function
- policy
- runtime
- CPU time
- peak memory

Learning metrics:
- episode returns
- moving average returns
- episode steps
- success rate
- trap rate
- TD errors
- MSE vs planning baseline
- policy agreement vs Value Iteration
- runtime
- CPU time
- peak memory
- environment steps

## Comparison rules

Correct comparisons:
- Policy Evaluation vs TD(0)/TD(lambda)
- Value Iteration/Policy Iteration/LP vs Q-learning/SARSA
- TD(0) vs TD(lambda)
- SARSA vs Q-learning
- Policy Iteration vs Value Iteration vs Linear Programming

Avoid misleading comparisons:
- Do not say Q-learning is worse than Value Iteration only because it is slower. Planning knows the model; Learning uses samples.
- Do not compare TD(0) directly with Value Iteration as if both solve the same task.
