---
name: rl-learning-algorithms
description: Use this skill when implementing, reviewing, testing, or explaining Learning Algorithms for the Grid-world project: TD(0), SARSA, Q-learning, and TD(lambda).
---

# RL Learning Algorithms Skill

## Scope

Use this skill for sample-based algorithms that do not directly use the full model P(s'|s,a). These algorithms learn from transitions: (S_t, A_t, R_{t+1}, S_{t+1}).

Algorithms:
- TD(0)
- SARSA
- Q-learning
- TD(lambda)

## Required environment interface

Learning algorithms should use:
- env.reset()
- env.step(action)
- env.get_actions(state)
- env.is_terminal(state)

Do not call env.transition_prob() inside learning algorithms, except in tests or evaluation comparisons.

## Mathematical requirements

TD(0):
delta_t = R_{t+1} + gamma V(S_{t+1}) - V(S_t)
V(S_t) <- V(S_t) + alpha delta_t

SARSA:
Q(S_t,A_t) <- Q(S_t,A_t) + alpha [R_{t+1} + gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]

Q-learning:
Q(S_t,A_t) <- Q(S_t,A_t) + alpha [R_{t+1} + gamma max_a' Q(S_{t+1},a') - Q(S_t,A_t)]

TD(lambda):
e_t(s) = gamma lambda e_{t-1}(s) + 1{S_t=s}
V(s) <- V(s) + alpha delta_t e_t(s)

## Implementation expectations

Create class-based implementations such as:
- TDZero
- TDLambda
- SARSA
- QLearning

Each class should expose:
- train()
- evaluate_policy()
- get_value_function()
- get_q_table() when applicable
- get_policy()
- get_metrics()

## Required learning metrics

For TD(0) and TD(lambda):
- value_function
- mean_absolute_td_error_per_episode
- mse_vs_policy_evaluation
- environment_steps
- episodes
- runtime_sec
- cpu_time_sec
- peak_memory_mb

For SARSA and Q-learning:
- q_table
- learned_policy
- episode_returns
- moving_average_returns
- episode_steps
- success_rate
- trap_rate
- average_steps
- td_errors
- policy_agreement_vs_value_iteration
- mse_vs_value_iteration, using V_learned(s)=max_a Q(s,a)
- runtime_sec
- cpu_time_sec
- peak_memory_mb
- q_updates
- environment_steps

## Comparisons

Compare inside the Learning group:
1. TD(0) vs TD(lambda):
   - MSE vs Policy Evaluation baseline
   - TD error curve
   - convergence speed
   - runtime and environment steps

2. SARSA vs Q-learning:
   - average return
   - success rate
   - trap rate
   - average steps
   - learned policy arrows
   - policy agreement with Value Iteration
   - stability across random seeds

## Important conceptual rules

- TD(0) and TD(lambda) are prediction/evaluation algorithms when used with a fixed policy.
- SARSA and Q-learning are control algorithms.
- SARSA is on-policy.
- Q-learning is off-policy.
- Do not compare TD(0) directly with Value Iteration as if they solve the same task.
