---
name: rl-testing-quality
description: Use this skill when adding tests, validating algorithm correctness, checking numerical stability, or preparing the project for reproducible grading.
---

# RL Testing and Quality Skill

## Scope

Use this skill for:
- unit tests
- sanity checks
- numerical correctness
- reproducibility
- final grading readiness

## Minimal test categories

Environment tests:
- reset() returns the configured start state
- step() respects walls and boundaries
- terminal states end episodes
- rewards match the reward configuration
- get_transitions() probabilities sum to 1
- deterministic transition has exactly one probability-1 outcome

Planning tests:
- Policy Evaluation converges below theta
- Value Iteration Bellman residual decreases
- Policy Iteration returns a stable policy
- Linear Programming value is close to Value Iteration value
- Policy agreement between PI and VI is high or 100% for deterministic Grid-world

Learning tests:
- Q-table shape is correct
- Q-learning updates the visited state-action pair
- SARSA uses the actual next action
- TD(0) updates only the current state
- TD(lambda) updates eligible previous states
- training metrics contain expected keys

Reproducibility tests:
- same seed gives same deterministic results
- JSON logs are valid
- generated figures are saved
- run_experiments.py completes without manual notebook execution

## Numerical tolerances

Use reasonable tolerances:
- value equality across planning methods: atol around 1e-5 to 1e-3 depending on theta
- residual threshold: theta such as 1e-6
- stochastic learning metrics should be evaluated over multiple seeds

## Final readiness checklist

Before considering a task complete:
- run tests
- run a small experiment
- check logs generated
- check figures generated
- confirm README instructions still work
- avoid committing large generated logs unless requested
