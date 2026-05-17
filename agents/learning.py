"""Skeletons for sample-based reinforcement learning algorithms."""

from __future__ import annotations

from typing import Any, TypeAlias

from envs.grid_world import Action, State
from envs.learning_grid_world import LearningGridWorld

Policy: TypeAlias = dict[State, Action | dict[Action, float]]
QTable: TypeAlias = dict[tuple[State, Action], float]
ValueFunction: TypeAlias = dict[State, float]


class TDZero:
    """TD(0) policy evaluation using temporal-difference error.

    delta = R + gamma V(S') - V(S)
    """

    def __init__(
        self,
        env: LearningGridWorld,
        alpha: float = 0.1,
        gamma: float | None = None,
        episodes: int = 1_000,
    ) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma if gamma is not None else env.gamma
        self.episodes = episodes
        self.values: ValueFunction = {state: 0.0 for state in env.get_states()}
        self.policy: Policy = {}
        self.metrics: dict[str, Any] = {}

    def train(self) -> ValueFunction:
        """Train a state-value function from sampled transitions.

        TODO: Implement TD(0) updates under a fixed behavior policy.
        """
        self.metrics["status"] = "not_implemented"
        return self.values

    def evaluate_policy(self) -> dict[str, Any]:
        """Evaluate the current policy.

        TODO: Add rollout-based evaluation metrics.
        """
        return {}

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
    """TD(lambda) policy evaluation with eligibility traces.

    e_t(s) = gamma lambda e_{t-1}(s) + 1{S_t=s}
    """

    def __init__(
        self,
        env: LearningGridWorld,
        alpha: float = 0.1,
        gamma: float | None = None,
        lambda_: float = 0.9,
        episodes: int = 1_000,
    ) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma if gamma is not None else env.gamma
        self.lambda_ = lambda_
        self.episodes = episodes
        self.values: ValueFunction = {state: 0.0 for state in env.get_states()}
        self.policy: Policy = {}
        self.metrics: dict[str, Any] = {}

    def train(self) -> ValueFunction:
        """Train with eligibility traces.

        TODO: Implement accumulating or replacing eligibility traces.
        """
        self.metrics["status"] = "not_implemented"
        return self.values

    def evaluate_policy(self) -> dict[str, Any]:
        """Evaluate the current policy with sampled episodes."""
        return {}

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
    """On-policy control with SARSA updates.

    Q(S,A) <- Q(S,A) + alpha[R + gamma Q(S',A') - Q(S,A)]
    """

    def __init__(
        self,
        env: LearningGridWorld,
        alpha: float = 0.1,
        gamma: float | None = None,
        epsilon: float = 0.1,
        episodes: int = 1_000,
    ) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma if gamma is not None else env.gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table: QTable = {(state, action): 0.0 for state in env.get_states() for action in env.get_actions(state)}
        self.policy: Policy = {}
        self.metrics: dict[str, Any] = {}

    def train(self) -> QTable:
        """Train an on-policy action-value function.

        TODO: Implement epsilon-greedy action selection and SARSA updates.
        """
        self.metrics["status"] = "not_implemented"
        return self.q_table

    def evaluate_policy(self) -> dict[str, Any]:
        """Evaluate the learned policy with sampled episodes."""
        return {}

    def get_q_table(self) -> QTable:
        """Return the learned action-value table."""
        return self.q_table

    def get_policy(self) -> Policy:
        """Return the learned on-policy control policy."""
        return self.policy

    def get_metrics(self) -> dict[str, Any]:
        """Return learning metrics."""
        return self.metrics


class QLearning:
    """Off-policy control with Q-learning updates.

    Q(S,A) <- Q(S,A) + alpha[R + gamma max_a' Q(S',a') - Q(S,A)]
    """

    def __init__(
        self,
        env: LearningGridWorld,
        alpha: float = 0.1,
        gamma: float | None = None,
        epsilon: float = 0.1,
        episodes: int = 1_000,
    ) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma if gamma is not None else env.gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table: QTable = {(state, action): 0.0 for state in env.get_states() for action in env.get_actions(state)}
        self.policy: Policy = {}
        self.metrics: dict[str, Any] = {}

    def train(self) -> QTable:
        """Train an off-policy action-value function.

        TODO: Implement epsilon-greedy behavior and greedy target updates.
        """
        self.metrics["status"] = "not_implemented"
        return self.q_table

    def evaluate_policy(self) -> dict[str, Any]:
        """Evaluate the greedy policy induced by the Q-table."""
        return {}

    def get_q_table(self) -> QTable:
        """Return the learned action-value table."""
        return self.q_table

    def get_policy(self) -> Policy:
        """Return the greedy policy induced by Q-learning."""
        return self.policy

    def get_metrics(self) -> dict[str, Any]:
        """Return learning metrics."""
        return self.metrics
