---
name: rl-planning-algorithms
description: "Use this skill when implementing, reviewing, testing, or explaining Planning Algorithms for the Grid-world project: Policy Evaluation, Policy Iteration, Value Iteration, and Linear Programming."
---

# RL Planning Algorithms Skill

## Scope

Use this skill for model-based algorithms that assume known MDP dynamics P(s'|s,a) and reward r(s,a,s').

Algorithms:
- Policy Evaluation
- Policy Iteration
- Value Iteration
- Linear Programming

## Required environment interface

Planning algorithms should use:
- env.get_states()
- env.get_actions(state)
- env.get_transitions(state, action)

Expected transition tuple:
(probability, next_state, reward, done)

## Mathematical requirements

Policy Evaluation:
V^pi(s) = sum_a pi(a|s) sum_s' P(s'|s,a)[r(s,a,s') + gamma V^pi(s')]

Value Iteration:
V_{k+1}(s) = max_a sum_s' P(s'|s,a)[r(s,a,s') + gamma V_k(s')]

Policy Improvement:
pi_new(s) = argmax_a sum_s' P(s'|s,a)[r(s,a,s') + gamma V^pi(s')]

Linear Programming:
minimize sum_s V(s)
subject to: V(s) >= r(s,a) + gamma sum_s' P(s'|s,a)V(s') for all s,a

## Implementation expectations

Create class-based implementations such as:
- PolicyEvaluation
- PolicyIteration
- ValueIteration
- LinearProgrammingPlanner

Each class should expose:
- run()
- get_value_function()
- get_policy()
- get_metrics()

## Metrics to collect

For Policy Evaluation:
- value_function
- bellman_residuals
- iterations
- runtime_sec
- cpu_time_sec
- peak_memory_mb
- bellman_backups

For Policy Iteration:
- final_policy
- value_function
- policy_improvement_iterations
- total_policy_evaluation_iterations
- policy_changes_per_iteration
- runtime_sec
- peak_memory_mb

For Value Iteration:
- optimal_value_function
- optimal_policy
- bellman_residuals
- iterations
- bellman_backups
- runtime_sec
- peak_memory_mb

For Linear Programming:
- optimal_value_function
- derived_policy
- number_of_variables
- number_of_constraints
- runtime_sec
- peak_memory_mb
- value_error_vs_value_iteration

## Comparisons

Compare inside the Planning group:
1. Policy Iteration vs Value Iteration vs Linear Programming:
   - infinity norm error between value functions
   - policy agreement
   - runtime
   - peak memory
   - iterations/backups/constraints

2. Policy Evaluation is not an optimal-control algorithm. Compare it against TD(0) and TD(lambda), not directly against Value Iteration.

## Visualization outputs

Save figures to report/figures/planning/:
- V^pi heatmap
- V* heatmap
- policy arrows
- Bellman residual curve
- policy changes curve
- planning runtime comparison
