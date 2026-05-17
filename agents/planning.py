"""Skeletons for model-based planning algorithms."""

from __future__ import annotations

from typing import Any, TypeAlias

from envs.grid_world import Action, State
from envs.planning_grid_world import PlanningGridWorld

Policy: TypeAlias = dict[State, Action | dict[Action, float]]
ValueFunction: TypeAlias = dict[State, float]


class PolicyEvaluation:
    """Evaluate a fixed policy with the Bellman expectation equation.

    V^pi(s) = sum_a pi(a|s) sum_s' P(s'|s,a)
              [r(s,a,s') + gamma V^pi(s')]
    """

    def __init__(
        self,
        env: PlanningGridWorld,
        policy: Policy | None = None,
        theta: float = 1e-6,
        max_iterations: int = 1_000,
    ) -> None:
        self.env = env
        self.policy = policy or {}
        self.theta = theta
        self.max_iterations = max_iterations
        self.values: ValueFunction = {state: 0.0 for state in env.get_states()}
        self.metrics: dict[str, Any] = {}

    def run(self) -> ValueFunction:
        """Run policy evaluation.

        TODO: Implement iterative Bellman expectation updates.
        """
        self.metrics["status"] = "not_implemented"
        return self.values

    def get_value_function(self) -> ValueFunction:
        """Return the current value-function estimate."""
        return self.values

    def get_policy(self) -> Policy:
        """Return the policy being evaluated."""
        return self.policy

    def get_metrics(self) -> dict[str, Any]:
        """Return training or convergence metrics."""
        return self.metrics


class PolicyIteration:
    """Solve an MDP using policy evaluation and policy improvement."""

    def __init__(
        self,
        env: PlanningGridWorld,
        theta: float = 1e-6,
        max_iterations: int = 1_000,
    ) -> None:
        self.env = env
        self.theta = theta
        self.max_iterations = max_iterations
        self.values: ValueFunction = {state: 0.0 for state in env.get_states()}
        self.policy: Policy = {}
        self.metrics: dict[str, Any] = {}

    def run(self) -> tuple[ValueFunction, Policy]:
        """Run policy evaluation followed by policy improvement.

        TODO: Add alternating policy evaluation and greedy improvement.
        """
        self.metrics["status"] = "not_implemented"
        return self.values, self.policy

    def get_value_function(self) -> ValueFunction:
        """Return the learned value function."""
        return self.values

    def get_policy(self) -> Policy:
        """Return the improved policy."""
        return self.policy

    def get_metrics(self) -> dict[str, Any]:
        """Return convergence metrics."""
        return self.metrics


class ValueIteration:
    """Solve an MDP using Bellman optimality backups.

    V_{k+1}(s) = max_a sum_s' P(s'|s,a)
                 [r(s,a,s') + gamma V_k(s')]
    """

    def __init__(
        self,
        env: PlanningGridWorld,
        theta: float = 1e-6,
        max_iterations: int = 1_000,
    ) -> None:
        self.env = env
        self.theta = theta
        self.max_iterations = max_iterations
        self.values: ValueFunction = {state: 0.0 for state in env.get_states()}
        self.policy: Policy = {}
        self.metrics: dict[str, Any] = {}

    def run(self) -> tuple[ValueFunction, Policy]:
        """Run value iteration.

        TODO: Implement Bellman optimality updates and extract a greedy policy.
        """
        self.metrics["status"] = "not_implemented"
        return self.values, self.policy

    def get_value_function(self) -> ValueFunction:
        """Return the optimal value-function estimate."""
        return self.values

    def get_policy(self) -> Policy:
        """Return a greedy policy induced by the value function."""
        return self.policy

    def get_metrics(self) -> dict[str, Any]:
        """Return convergence metrics."""
        return self.metrics


class LinearProgrammingPlanner:
    """Solve an MDP by formulating Bellman optimality as linear constraints.

    minimize sum_s V(s)
    subject to V(s) >= r(s,a) + gamma sum_s' P(s'|s,a)V(s')
    """

    def __init__(self, env: PlanningGridWorld, solver_options: dict[str, Any] | None = None) -> None:
        self.env = env
        self.solver_options = solver_options or {}
        self.values: ValueFunction = {state: 0.0 for state in env.get_states()}
        self.policy: Policy = {}
        self.metrics: dict[str, Any] = {}

    def run(self) -> tuple[ValueFunction, Policy]:
        """Run the linear-programming planner.

        TODO: Use scipy.optimize.linprog to solve the constrained problem.
        """
        self.metrics["status"] = "not_implemented"
        return self.values, self.policy

    def get_value_function(self) -> ValueFunction:
        """Return the LP value-function solution."""
        return self.values

    def get_policy(self) -> Policy:
        """Return a greedy policy induced by the LP value function."""
        return self.policy

    def get_metrics(self) -> dict[str, Any]:
        """Return solver and convergence metrics."""
        return self.metrics
