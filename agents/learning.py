"""Sample-based reinforcement learning algorithms for Grid-world."""

from __future__ import annotations

import random
import time
import tracemalloc
from collections.abc import Mapping
from typing import Any, TypeAlias

from envs.grid_world import Action, State
from envs.learning_grid_world import LearningGridWorld

Policy: TypeAlias = dict[State, Action | dict[Action, float]]
QTable: TypeAlias = dict[tuple[State, Action], float]
ValueFunction: TypeAlias = dict[State, float]


class TDZero:
    """TD(0) policy evaluation using sampled transitions.

    The learner estimates V^pi under a fixed behavior policy. If no policy is
    provided, it uses a uniform random policy over valid actions.
    """

    def __init__(
        self,
        env: LearningGridWorld,
        alpha: float = 0.1,
        gamma: float | None = None,
        episodes: int = 1_000,
        max_steps_per_episode: int | None = None,
        policy: Policy | None = None,
        seed: int | None = None,
    ) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma if gamma is not None else env.gamma
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode or _default_max_steps(env)
        self.policy = policy or {}
        self.values: ValueFunction = {state: 0.0 for state in env.get_states()}
        self.metrics: dict[str, Any] = {}
        self._rng = random.Random(seed if seed is not None else env.seed)

    def train(self) -> ValueFunction:
        """Train a state-value function from sampled transitions."""
        with _profile_block() as profile:
            mean_abs_td_errors: list[float] = []
            environment_steps = 0

            for _ in range(self.episodes):
                state = self.env.reset()
                episode_td_errors: list[float] = []

                for _ in range(self.max_steps_per_episode):
                    action = _sample_policy_action(self.policy, self.env, state, self._rng)
                    next_state, reward, done, _ = self.env.step(action)

                    target = reward if done else reward + self.gamma * self.values[next_state]
                    td_error = target - self.values[state]
                    self.values[state] += self.alpha * td_error

                    episode_td_errors.append(abs(td_error))
                    environment_steps += 1
                    state = next_state
                    if done:
                        break

                mean_abs_td_errors.append(_mean(episode_td_errors))

        self.metrics = {
            "algorithm": "td_zero",
            "episodes": self.episodes,
            "environment_steps": environment_steps,
            "mean_absolute_td_error_per_episode": mean_abs_td_errors,
            "final_mean_absolute_td_error": mean_abs_td_errors[-1] if mean_abs_td_errors else 0.0,
            "mse_vs_policy_evaluation": None,
            "value_function": _serialize_state_map(self.values),
            **profile(),
        }
        return self.values

    def evaluate_policy(self, episodes: int = 100) -> dict[str, Any]:
        """Evaluate the fixed behavior policy with sampled rollouts."""
        return _evaluate_state_policy(
            self.env,
            self.policy,
            self._rng,
            episodes,
            self.max_steps_per_episode,
        )

    def get_value_function(self) -> ValueFunction:
        """Return the learned state-value function."""
        return self.values

    def get_policy(self) -> Policy:
        """Return the policy used by TD evaluation."""
        return self.policy

    def get_metrics(self) -> dict[str, Any]:
        """Return learning metrics."""
        return self.metrics


class TDLambda:
    """TD(lambda) policy evaluation with accumulating eligibility traces."""

    def __init__(
        self,
        env: LearningGridWorld,
        alpha: float = 0.1,
        gamma: float | None = None,
        lambda_: float = 0.9,
        episodes: int = 1_000,
        max_steps_per_episode: int | None = None,
        policy: Policy | None = None,
        seed: int | None = None,
    ) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma if gamma is not None else env.gamma
        self.lambda_ = lambda_
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode or _default_max_steps(env)
        self.policy = policy or {}
        self.values: ValueFunction = {state: 0.0 for state in env.get_states()}
        self.metrics: dict[str, Any] = {}
        self._rng = random.Random(seed if seed is not None else env.seed)

    def train(self) -> ValueFunction:
        """Train a state-value function using eligibility traces."""
        with _profile_block() as profile:
            mean_abs_td_errors: list[float] = []
            environment_steps = 0
            states = self.env.get_states()

            for _ in range(self.episodes):
                state = self.env.reset()
                traces: dict[State, float] = {trace_state: 0.0 for trace_state in states}
                episode_td_errors: list[float] = []

                for _ in range(self.max_steps_per_episode):
                    action = _sample_policy_action(self.policy, self.env, state, self._rng)
                    next_state, reward, done, _ = self.env.step(action)

                    target = reward if done else reward + self.gamma * self.values[next_state]
                    td_error = target - self.values[state]
                    traces[state] += 1.0

                    for trace_state in states:
                        self.values[trace_state] += self.alpha * td_error * traces[trace_state]
                        traces[trace_state] *= self.gamma * self.lambda_

                    episode_td_errors.append(abs(td_error))
                    environment_steps += 1
                    state = next_state
                    if done:
                        break

                mean_abs_td_errors.append(_mean(episode_td_errors))

        self.metrics = {
            "algorithm": "td_lambda",
            "episodes": self.episodes,
            "lambda": self.lambda_,
            "environment_steps": environment_steps,
            "mean_absolute_td_error_per_episode": mean_abs_td_errors,
            "final_mean_absolute_td_error": mean_abs_td_errors[-1] if mean_abs_td_errors else 0.0,
            "mse_vs_policy_evaluation": None,
            "value_function": _serialize_state_map(self.values),
            **profile(),
        }
        return self.values

    def evaluate_policy(self, episodes: int = 100) -> dict[str, Any]:
        """Evaluate the fixed behavior policy with sampled rollouts."""
        return _evaluate_state_policy(
            self.env,
            self.policy,
            self._rng,
            episodes,
            self.max_steps_per_episode,
        )

    def get_value_function(self) -> ValueFunction:
        """Return the learned state-value function."""
        return self.values

    def get_policy(self) -> Policy:
        """Return the policy used by TD(lambda)."""
        return self.policy

    def get_metrics(self) -> dict[str, Any]:
        """Return learning metrics."""
        return self.metrics


class SARSA:
    """On-policy control with SARSA updates."""

    def __init__(
        self,
        env: LearningGridWorld,
        alpha: float = 0.1,
        gamma: float | None = None,
        epsilon: float = 0.1,
        episodes: int = 1_000,
        max_steps_per_episode: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma if gamma is not None else env.gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode or _default_max_steps(env)
        self.q_table: QTable = _initial_q_table(env)
        self.policy: Policy = {}
        self.metrics: dict[str, Any] = {}
        self._rng = random.Random(seed if seed is not None else env.seed)

    def train(self) -> QTable:
        """Train an on-policy action-value function."""
        with _profile_block() as profile:
            episode_returns: list[float] = []
            episode_steps: list[int] = []
            td_errors: list[float] = []
            outcomes: list[dict[str, str]] = []
            environment_steps = 0
            q_updates = 0

            for _ in range(self.episodes):
                state = self.env.reset()
                action = _epsilon_greedy_action(self.env, self.q_table, state, self.epsilon, self._rng)
                total_return = 0.0

                for step_index in range(1, self.max_steps_per_episode + 1):
                    next_state, reward, done, _ = self.env.step(action)
                    total_return += reward

                    if done:
                        target = reward
                    else:
                        next_action = _epsilon_greedy_action(
                            self.env,
                            self.q_table,
                            next_state,
                            self.epsilon,
                            self._rng,
                        )
                        target = reward + self.gamma * self.q_table[(next_state, next_action)]

                    td_error = target - self.q_table[(state, action)]
                    self.q_table[(state, action)] += self.alpha * td_error
                    td_errors.append(td_error)
                    environment_steps += 1
                    q_updates += 1

                    if done:
                        break
                    state = next_state
                    action = next_action

                episode_returns.append(total_return)
                episode_steps.append(step_index)
                outcomes.append({"outcome": _episode_outcome(self.env, next_state)})

        self.policy = _greedy_policy(self.env, self.q_table)
        self.metrics = _control_metrics(
            "sarsa",
            self.env,
            self.q_table,
            self.policy,
            episode_returns,
            episode_steps,
            outcomes,
            td_errors,
            environment_steps,
            q_updates,
            profile(),
        )
        return self.q_table

    def evaluate_policy(self, episodes: int = 100) -> dict[str, Any]:
        """Evaluate the learned greedy policy with sampled episodes."""
        return _evaluate_state_policy(
            self.env,
            self.policy,
            self._rng,
            episodes,
            self.max_steps_per_episode,
        )

    def get_q_table(self) -> QTable:
        """Return the learned action-value table."""
        return self.q_table

    def get_value_function(self) -> ValueFunction:
        """Return V(s)=max_a Q(s,a) induced by the learned Q-table."""
        return _value_from_q_table(self.env, self.q_table)

    def get_policy(self) -> Policy:
        """Return the learned on-policy control policy."""
        return self.policy

    def get_metrics(self) -> dict[str, Any]:
        """Return learning metrics."""
        return self.metrics


class QLearning:
    """Off-policy control with Q-learning updates."""

    def __init__(
        self,
        env: LearningGridWorld,
        alpha: float = 0.1,
        gamma: float | None = None,
        epsilon: float = 0.1,
        episodes: int = 1_000,
        max_steps_per_episode: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma if gamma is not None else env.gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode or _default_max_steps(env)
        self.q_table: QTable = _initial_q_table(env)
        self.policy: Policy = {}
        self.metrics: dict[str, Any] = {}
        self._rng = random.Random(seed if seed is not None else env.seed)

    def train(self) -> QTable:
        """Train an off-policy action-value function."""
        with _profile_block() as profile:
            episode_returns: list[float] = []
            episode_steps: list[int] = []
            td_errors: list[float] = []
            outcomes: list[dict[str, str]] = []
            environment_steps = 0
            q_updates = 0

            for _ in range(self.episodes):
                state = self.env.reset()
                total_return = 0.0

                for step_index in range(1, self.max_steps_per_episode + 1):
                    action = _epsilon_greedy_action(self.env, self.q_table, state, self.epsilon, self._rng)
                    next_state, reward, done, _ = self.env.step(action)
                    total_return += reward

                    target = reward
                    if not done:
                        next_actions = self.env.get_actions(next_state)
                        best_next_q = max(self.q_table[(next_state, next_action)] for next_action in next_actions)
                        target += self.gamma * best_next_q

                    td_error = target - self.q_table[(state, action)]
                    self.q_table[(state, action)] += self.alpha * td_error
                    td_errors.append(td_error)
                    environment_steps += 1
                    q_updates += 1

                    state = next_state
                    if done:
                        break

                episode_returns.append(total_return)
                episode_steps.append(step_index)
                outcomes.append({"outcome": _episode_outcome(self.env, state)})

        self.policy = _greedy_policy(self.env, self.q_table)
        self.metrics = _control_metrics(
            "q_learning",
            self.env,
            self.q_table,
            self.policy,
            episode_returns,
            episode_steps,
            outcomes,
            td_errors,
            environment_steps,
            q_updates,
            profile(),
        )
        return self.q_table

    def evaluate_policy(self, episodes: int = 100) -> dict[str, Any]:
        """Evaluate the learned greedy policy with sampled episodes."""
        return _evaluate_state_policy(
            self.env,
            self.policy,
            self._rng,
            episodes,
            self.max_steps_per_episode,
        )

    def get_q_table(self) -> QTable:
        """Return the learned action-value table."""
        return self.q_table

    def get_value_function(self) -> ValueFunction:
        """Return V(s)=max_a Q(s,a) induced by the learned Q-table."""
        return _value_from_q_table(self.env, self.q_table)

    def get_policy(self) -> Policy:
        """Return the greedy policy induced by Q-learning."""
        return self.policy

    def get_metrics(self) -> dict[str, Any]:
        """Return learning metrics."""
        return self.metrics


class _profile_block:
    """Context manager for lightweight runtime and memory metrics."""

    def __enter__(self) -> Any:
        tracemalloc.start()
        self._wall_start = time.perf_counter()
        self._cpu_start = time.process_time()
        return self._result

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        self._metrics = {
            "runtime_sec": time.perf_counter() - self._wall_start,
            "cpu_time_sec": time.process_time() - self._cpu_start,
            "current_memory_mb": current_bytes / (1024 * 1024),
            "peak_memory_mb": peak_bytes / (1024 * 1024),
        }
        tracemalloc.stop()

    def _result(self) -> dict[str, float]:
        return self._metrics


def _default_max_steps(env: LearningGridWorld) -> int:
    """Return a conservative episode cap for an 8x8 navigation task."""
    rows, cols = env.grid_size
    return rows * cols * 4


def _initial_q_table(env: LearningGridWorld) -> QTable:
    """Create a zero-filled action-value table for all non-terminal states."""
    return {
        (state, action): 0.0
        for state in env.get_states()
        for action in env.get_actions(state)
    }


def _epsilon_greedy_action(
    env: LearningGridWorld,
    q_table: QTable,
    state: State,
    epsilon: float,
    rng: random.Random,
) -> Action:
    """Choose an action with epsilon-greedy exploration."""
    actions = list(env.get_actions(state))
    if not actions:
        raise ValueError(f"No available actions for state {state}.")
    if rng.random() < epsilon:
        return rng.choice(actions)
    return _best_action(actions, q_table, state)


def _best_action(actions: list[Action], q_table: QTable, state: State) -> Action:
    """Return the greedy action using action order for deterministic ties."""
    return max(actions, key=lambda action: q_table[(state, action)])


def _sample_policy_action(
    policy: Policy,
    env: LearningGridWorld,
    state: State,
    rng: random.Random,
) -> Action:
    """Sample an action from a deterministic, stochastic, or uniform policy."""
    actions = list(env.get_actions(state))
    if not actions:
        raise ValueError(f"No available actions for state {state}.")

    policy_entry = policy.get(state)
    if isinstance(policy_entry, str):
        return policy_entry
    if isinstance(policy_entry, Mapping):
        threshold = rng.random()
        cumulative = 0.0
        fallback_action = actions[-1]
        for action, probability in policy_entry.items():
            fallback_action = action
            cumulative += probability
            if threshold <= cumulative:
                return action
        return fallback_action
    return rng.choice(actions)


def _greedy_policy(env: LearningGridWorld, q_table: QTable) -> Policy:
    """Extract a deterministic greedy policy from an action-value table."""
    policy: Policy = {}
    for state in env.get_states():
        actions = list(env.get_actions(state))
        if actions:
            policy[state] = _best_action(actions, q_table, state)
    return policy


def _value_from_q_table(env: LearningGridWorld, q_table: QTable) -> ValueFunction:
    """Compute state values induced by a Q-table."""
    values: ValueFunction = {}
    for state in env.get_states():
        actions = env.get_actions(state)
        values[state] = max((q_table[(state, action)] for action in actions), default=0.0)
    return values


def _evaluate_state_policy(
    env: LearningGridWorld,
    policy: Policy,
    rng: random.Random,
    episodes: int,
    max_steps_per_episode: int,
) -> dict[str, Any]:
    """Roll out a policy and summarize returns, steps, and outcomes."""
    episode_returns: list[float] = []
    episode_steps: list[int] = []
    outcomes: list[dict[str, str]] = []

    for _ in range(episodes):
        state = env.reset()
        total_return = 0.0
        for step_index in range(1, max_steps_per_episode + 1):
            action = _sample_policy_action(policy, env, state, rng)
            state, reward, done, _ = env.step(action)
            total_return += reward
            if done:
                break
        episode_returns.append(total_return)
        episode_steps.append(step_index)
        outcomes.append({"outcome": _episode_outcome(env, state)})

    return {
        "episodes": episodes,
        "average_return": _mean(episode_returns),
        "average_steps": _mean(episode_steps),
        "success_rate": _rate(outcomes, "success"),
        "trap_rate": _rate(outcomes, "trap"),
    }


def _control_metrics(
    algorithm: str,
    env: LearningGridWorld,
    q_table: QTable,
    policy: Policy,
    episode_returns: list[float],
    episode_steps: list[int],
    outcomes: list[dict[str, str]],
    td_errors: list[float],
    environment_steps: int,
    q_updates: int,
    profile_metrics: dict[str, float],
) -> dict[str, Any]:
    """Build JSON-safe metrics for a control learner."""
    return {
        "algorithm": algorithm,
        "episodes": len(episode_returns),
        "episode_returns": episode_returns,
        "average_return": _mean(episode_returns),
        "moving_average_returns": _moving_average(episode_returns),
        "episode_steps": episode_steps,
        "success_rate": _rate(outcomes, "success"),
        "trap_rate": _rate(outcomes, "trap"),
        "average_steps": _mean(episode_steps),
        "td_errors": td_errors,
        "final_mean_absolute_td_error": _mean([abs(error) for error in td_errors[-100:]]),
        "policy_agreement_vs_value_iteration": None,
        "mse_vs_value_iteration": None,
        "environment_steps": environment_steps,
        "q_updates": q_updates,
        "q_table": _serialize_q_table(q_table),
        "learned_policy": _serialize_policy(policy),
        "value_function": _serialize_state_map(_value_from_q_table(env, q_table)),
        **profile_metrics,
    }


def _episode_outcome(env: LearningGridWorld, state: State) -> str:
    """Classify an episode's terminal or capped ending."""
    if state in env.goal_states:
        return "success"
    if state in env.trap_states:
        return "trap"
    return "timeout"


def _moving_average(values: list[float], window: int = 25) -> list[float]:
    """Return a trailing moving average series."""
    averages: list[float] = []
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        averages.append(_mean(values[start : index + 1]))
    return averages


def _rate(outcomes: list[dict[str, str]], target: str) -> float:
    """Compute the fraction of outcomes matching a label."""
    if not outcomes:
        return 0.0
    return sum(1 for outcome in outcomes if outcome["outcome"] == target) / len(outcomes)


def _mean(values: list[float] | list[int]) -> float:
    """Return the arithmetic mean or zero for an empty list."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _serialize_state_map(values: Mapping[State, float]) -> dict[str, float]:
    """Serialize state-keyed maps for JSON logs."""
    return {_state_key(state): float(value) for state, value in values.items()}


def _serialize_q_table(q_table: QTable) -> dict[str, float]:
    """Serialize a Q-table for JSON logs."""
    return {
        f"{_state_key(state)}|{action}": float(value)
        for (state, action), value in q_table.items()
    }


def _serialize_policy(policy: Policy) -> dict[str, Any]:
    """Serialize a policy for JSON logs."""
    serialized: dict[str, Any] = {}
    for state, action in policy.items():
        serialized[_state_key(state)] = action
    return serialized


def _state_key(state: State) -> str:
    """Return a compact JSON object key for a grid state."""
    return f"{state[0]},{state[1]}"
