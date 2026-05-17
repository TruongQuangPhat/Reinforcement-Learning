---
name: rl-visualization-reporting
description: Use this skill when creating visualizations, notebook explanations, report figures, LaTeX content, or experiment analysis for the Grid-world RL project.
---

# RL Visualization and Reporting Skill

## Scope

Use this skill for:
- utils/visualization.py
- notebooks/planning.ipynb
- notebooks/learning.ipynb
- report figures
- result interpretation

## Required figures

Planning:
- fixed-policy arrows
- V^pi heatmap
- V* heatmap
- optimal policy arrows
- Bellman residual curve
- policy changes per iteration
- runtime comparison

Learning:
- episode return curve
- moving average return curve
- steps per episode
- success rate curve
- trap rate curve
- TD error curve
- learned policy arrows
- max_a Q(s,a) heatmap
- TD(0) vs TD(lambda) MSE curve
- SARSA vs Q-learning comparison plots

## Plot quality rules

- Every figure must have a title, axis labels, legend when needed, and caption in the report.
- Use consistent state coordinate convention: row, col.
- Mark Start, Goal, Trap, and Wall clearly.
- Save figures under:
  - report/figures/planning/
  - report/figures/learning/
  - report/figures/comparison/

## Notebook rules

notebooks/planning.ipynb should explain:
- MDP definition
- Policy Evaluation
- Policy Iteration
- Value Iteration
- Linear Programming
- planning metrics and comparisons

notebooks/learning.ipynb should explain:
- model-free learning setup
- TD error
- TD(0)
- TD(lambda)
- SARSA
- Q-learning
- learning metrics and comparisons

Do not put core algorithm implementation inside notebooks. Import from envs/, agents/, and utils/.

## Report interpretation rules

When writing analysis:
- Do not claim Learning is worse than Planning only because it is slower.
- Explain that Planning has access to P and r, while Learning uses samples.
- Compare Policy Evaluation with TD methods.
- Compare Value Iteration/Policy Iteration/LP with SARSA/Q-learning.
- Discuss runtime, peak RAM, backups, environment steps, and quality metrics separately.
