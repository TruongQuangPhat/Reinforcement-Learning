"""Model-based planning algorithms for finite Grid-world MDPs."""

from __future__ import annotations

from typing import Any, TypeAlias

from scipy.optimize import linprog

from envs.grid_world import Action, State
from envs.planning_grid_world import PlanningGridWorld
from utils.logging_utils import (
    format_scientific,
    log_message,
    print_progress_header,
    print_progress_row,
)

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
        verbose: int = 0,
        log_interval: int = 10,
    ) -> None:
        if theta <= 0.0:
            raise ValueError(f"theta must be positive, got {theta}")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")

        self.env = env
        self.policy = policy or self._uniform_random_policy()
        self.theta = theta
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.log_interval = _validated_log_interval(log_interval)
        self.values: ValueFunction = {state: 0.0 for state in env.get_states()}
        self.metrics: dict[str, Any] = {}

    def run(self) -> ValueFunction:
        """Run iterative policy evaluation until convergence or iteration cap."""
        bellman_residuals: list[float] = []
        bellman_backups = 0
        progress_header_printed = False

        for iteration in range(1, self.max_iterations + 1):
            old_values = dict(self.values)
            delta = 0.0

            for state in self.env.get_states():
                if self.env.is_terminal(state):
                    self.values[state] = 0.0
                    continue

                action_distribution = _policy_action_distribution(self.env, self.policy, state)
                new_value = 0.0
                for action, action_probability in action_distribution.items():
                    transitions = self.env.get_transitions(state, action)
                    for probability, next_state, reward, done in transitions:
                        bootstrap_value = 0.0 if done else old_values[next_state]
                        new_value += action_probability * probability * (
                            reward + self.env.gamma * bootstrap_value
                        )

                self.values[state] = new_value
                bellman_backups += 1
                delta = max(delta, abs(old_values[state] - new_value))

            bellman_residuals.append(delta)
            if iteration % self.log_interval == 0 or delta < self.theta:
                if not progress_header_printed:
                    _print_planning_progress_header("PolicyEvaluation", self.verbose)
                    progress_header_printed = True
                _print_planning_progress_row(iteration, delta, bellman_backups, self.verbose)
            if delta < self.theta:
                break

        status = (
            "converged"
            if bellman_residuals[-1] < self.theta
            else "max_iterations_reached"
        )
        self.metrics = {
            "status": status,
            "theta": self.theta,
            "gamma": self.env.gamma,
            "iterations": len(bellman_residuals),
            "bellman_residuals": bellman_residuals,
            "final_bellman_residual": bellman_residuals[-1],
            "bellman_backups": bellman_backups,
            "value_function": _serialize_value_function(self.values),
            "policy": _serialize_policy(self.policy),
        }
        return self.values

    def _uniform_random_policy(self) -> Policy:
        """Create a uniform random policy over available actions."""
        policy: Policy = {}
        for state in self.env.get_states():
            actions = self.env.get_actions(state)
            if actions:
                probability = 1.0 / len(actions)
                policy[state] = {action: probability for action in actions}
        return policy

    def _action_distribution(self, state: State) -> dict[Action, float]:
        """Return a validated action distribution for one state."""
        return _policy_action_distribution(self.env, self.policy, state)

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
        verbose: int = 0,
        log_interval: int = 10,
    ) -> None:
        if theta <= 0.0:
            raise ValueError(f"theta must be positive, got {theta}")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")

        self.env = env
        self.theta = theta
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.log_interval = _validated_log_interval(log_interval)
        self.values: ValueFunction = {state: 0.0 for state in env.get_states()}
        self.policy: Policy = _initial_deterministic_policy(env)
        self.metrics: dict[str, Any] = {}

    def run(self) -> tuple[ValueFunction, Policy]:
        """Run exact policy iteration with iterative policy evaluation."""
        policy_changes_per_iteration: list[int] = []
        evaluation_iterations_per_improvement: list[int] = []
        evaluation_residuals_per_improvement: list[list[float]] = []
        evaluation_converged_per_improvement: list[bool] = []
        total_evaluation_iterations = 0
        bellman_backups = 0
        policy_stable = False
        evaluation_converged = False
        progress_header_printed = False

        for improvement_iteration in range(1, self.max_iterations + 1):
            (
                eval_iterations,
                eval_residuals,
                eval_backups,
                evaluation_converged,
            ) = self._evaluate_current_policy()
            total_evaluation_iterations += eval_iterations
            bellman_backups += eval_backups
            evaluation_iterations_per_improvement.append(eval_iterations)
            evaluation_residuals_per_improvement.append(eval_residuals)
            evaluation_converged_per_improvement.append(evaluation_converged)

            policy_stable = True
            policy_changes = 0
            new_policy: Policy = {}

            for state in self.env.get_states():
                if self.env.is_terminal(state):
                    continue

                old_action = self.policy.get(state)
                new_action, _ = _greedy_action_and_value(self.env, state, self.values)
                new_policy[state] = new_action
                if old_action != new_action:
                    policy_stable = False
                    policy_changes += 1

            self.policy = new_policy
            policy_changes_per_iteration.append(policy_changes)
            should_log = (
                improvement_iteration % self.log_interval == 0
                or (policy_stable and evaluation_converged)
            )
            if should_log:
                final_residual = eval_residuals[-1] if eval_residuals else 0.0
                if not progress_header_printed:
                    _print_planning_progress_header("PolicyIteration", self.verbose)
                    progress_header_printed = True
                _print_planning_progress_row(
                    improvement_iteration,
                    final_residual,
                    bellman_backups,
                    self.verbose,
                )
            if policy_stable and evaluation_converged:
                break

        status = (
            "converged"
            if policy_stable and evaluation_converged
            else "max_iterations_reached"
        )
        self.metrics = {
            "status": status,
            "theta": self.theta,
            "gamma": self.env.gamma,
            "policy_stable": policy_stable,
            "final_policy_evaluation_converged": evaluation_converged,
            "final_policy_evaluation_residual": (
                evaluation_residuals_per_improvement[-1][-1]
                if evaluation_residuals_per_improvement[-1]
                else None
            ),
            "final_bellman_residual": (
                evaluation_residuals_per_improvement[-1][-1]
                if evaluation_residuals_per_improvement[-1]
                else None
            ),
            "policy_improvement_iterations": len(policy_changes_per_iteration),
            "total_policy_evaluation_iterations": total_evaluation_iterations,
            "policy_evaluation_iterations_per_improvement": (
                evaluation_iterations_per_improvement
            ),
            "policy_evaluation_converged_per_improvement": (
                evaluation_converged_per_improvement
            ),
            "policy_evaluation_residuals_per_improvement": (
                evaluation_residuals_per_improvement
            ),
            "policy_changes_per_iteration": policy_changes_per_iteration,
            "bellman_backups": bellman_backups,
            "value_function": _serialize_value_function(self.values),
            "final_policy": _serialize_policy(self.policy),
        }
        return self.values, self.policy

    def _evaluate_current_policy(self) -> tuple[int, list[float], int, bool]:
        """Evaluate the current deterministic policy in place."""
        residuals: list[float] = []
        bellman_backups = 0

        for iteration in range(1, self.max_iterations + 1):
            old_values = dict(self.values)
            delta = 0.0

            for state in self.env.get_states():
                if self.env.is_terminal(state):
                    self.values[state] = 0.0
                    continue

                new_value = _expected_policy_value(
                    self.env,
                    state,
                    _policy_action_distribution(self.env, self.policy, state),
                    old_values,
                )
                self.values[state] = new_value
                bellman_backups += 1
                delta = max(delta, abs(old_values[state] - new_value))

            residuals.append(delta)
            if delta < self.theta:
                return iteration, residuals, bellman_backups, True

        return self.max_iterations, residuals, bellman_backups, False

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
        verbose: int = 0,
        log_interval: int = 10,
    ) -> None:
        if theta <= 0.0:
            raise ValueError(f"theta must be positive, got {theta}")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")

        self.env = env
        self.theta = theta
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.log_interval = _validated_log_interval(log_interval)
        self.values: ValueFunction = {state: 0.0 for state in env.get_states()}
        self.policy: Policy = {}
        self.metrics: dict[str, Any] = {}

    def run(self) -> tuple[ValueFunction, Policy]:
        """Run Bellman optimality updates and extract a greedy policy."""
        bellman_residuals: list[float] = []
        bellman_backups = 0
        progress_header_printed = False

        for iteration in range(1, self.max_iterations + 1):
            old_values = dict(self.values)
            delta = 0.0

            for state in self.env.get_states():
                if self.env.is_terminal(state):
                    self.values[state] = 0.0
                    continue

                _, best_value = _greedy_action_and_value(self.env, state, old_values)
                self.values[state] = best_value
                bellman_backups += 1
                delta = max(delta, abs(old_values[state] - best_value))

            bellman_residuals.append(delta)
            if iteration % self.log_interval == 0 or delta < self.theta:
                if not progress_header_printed:
                    _print_planning_progress_header("ValueIteration", self.verbose)
                    progress_header_printed = True
                _print_planning_progress_row(iteration, delta, bellman_backups, self.verbose)
            if delta < self.theta:
                break

        self.policy = _greedy_policy(self.env, self.values)
        status = (
            "converged"
            if bellman_residuals[-1] < self.theta
            else "max_iterations_reached"
        )
        self.metrics = {
            "status": status,
            "theta": self.theta,
            "gamma": self.env.gamma,
            "iterations": len(bellman_residuals),
            "bellman_residuals": bellman_residuals,
            "final_bellman_residual": bellman_residuals[-1],
            "bellman_backups": bellman_backups,
            "optimal_value_function": _serialize_value_function(self.values),
            "optimal_policy": _serialize_policy(self.policy),
        }
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

    def __init__(
        self,
        env: PlanningGridWorld,
        solver_options: dict[str, Any] | None = None,
        verbose: int = 0,
        log_interval: int = 10,
    ) -> None:
        self.env = env
        self.solver_options = solver_options or {}
        self.verbose = verbose
        self.log_interval = _validated_log_interval(log_interval)
        self.values: ValueFunction = {state: 0.0 for state in env.get_states()}
        self.policy: Policy = {}
        self.metrics: dict[str, Any] = {}

    def run(self) -> tuple[ValueFunction, Policy]:
        """Solve Bellman optimality inequalities with linear programming."""
        states = self.env.get_states()
        state_to_index = {state: index for index, state in enumerate(states)}
        c = [1.0 for _ in states]
        a_ub: list[list[float]] = []
        b_ub: list[float] = []

        for state in states:
            if self.env.is_terminal(state):
                continue

            for action in self.env.get_actions(state):
                row = [0.0 for _ in states]
                row[state_to_index[state]] = -1.0
                expected_reward = 0.0
                for probability, next_state, reward, done in self.env.get_transitions(
                    state,
                    action,
                ):
                    expected_reward += probability * reward
                    if not done:
                        row[state_to_index[next_state]] += self.env.gamma * probability
                a_ub.append(row)
                b_ub.append(-expected_reward)

        bounds = [
            (0.0, 0.0) if self.env.is_terminal(state) else (None, None)
            for state in states
        ]
        solver_kwargs = {"method": "highs", **self.solver_options}
        log_message(
            (
                "[LinearProgrammingPlanner] "
                f"variables={len(states)} constraints={len(a_ub)}"
            ),
            self.verbose,
            min_verbose=2,
        )
        result = linprog(c=c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, **solver_kwargs)

        if not result.success:
            self.metrics = {
                "status": "solver_failed",
                "solver_status": int(result.status),
                "solver_message": result.message,
                "number_of_variables": len(states),
                "number_of_constraints": len(a_ub),
            }
            raise RuntimeError(f"Linear programming planner failed: {result.message}")

        self.values = {
            state: float(result.x[state_to_index[state]])
            for state in states
        }
        self.policy = _greedy_policy(self.env, self.values)
        self.metrics = {
            "status": "optimal",
            "gamma": self.env.gamma,
            "solver_status": int(result.status),
            "solver_message": result.message,
            "solver_iterations": int(getattr(result, "nit", 0)),
            "solver_iter": int(getattr(result, "nit", 0)),
            "objective_value": float(result.fun),
            "number_of_variables": len(states),
            "number_of_constraints": len(a_ub),
            "optimal_value_function": _serialize_value_function(self.values),
            "derived_policy": _serialize_policy(self.policy),
            "value_error_vs_value_iteration": None,
            "policy_agreement_vs_value_iteration": None,
        }
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


def _serialize_state(state: State) -> str:
    """Convert a grid state to a stable JSON object key."""
    return f"{state[0]},{state[1]}"


def _initial_deterministic_policy(env: PlanningGridWorld) -> Policy:
    """Create a stable deterministic initial policy."""
    policy: Policy = {}
    for state in env.get_states():
        actions = env.get_actions(state)
        if actions:
            policy[state] = actions[0]
    return policy


def _policy_action_distribution(
    env: PlanningGridWorld,
    policy: Policy,
    state: State,
) -> dict[Action, float]:
    """Return a validated action distribution for one state."""
    actions = env.get_actions(state)
    if not actions:
        return {}

    policy_entry = policy.get(state)
    if policy_entry is None:
        probability = 1.0 / len(actions)
        return {action: probability for action in actions}

    if isinstance(policy_entry, str):
        if policy_entry not in actions:
            raise ValueError(f"Policy action {policy_entry!r} is invalid for state {state}")
        return {policy_entry: 1.0}

    distribution: dict[Action, float] = {}
    for action, probability in policy_entry.items():
        if action not in actions:
            raise ValueError(f"Policy action {action!r} is invalid for state {state}")
        if probability < 0:
            raise ValueError(f"Policy probability must be non-negative, got {probability}")
        if probability > 0:
            distribution[action] = float(probability)

    total_probability = sum(distribution.values())
    if total_probability <= 0.0:
        raise ValueError(
            f"Policy probabilities for state {state} must sum to a positive value"
        )
    if abs(total_probability - 1.0) > 1e-9:
        raise ValueError(
            f"Policy probabilities for state {state} must sum to 1.0, "
            f"got {total_probability}"
        )

    return distribution


def _expected_policy_value(
    env: PlanningGridWorld,
    state: State,
    action_distribution: dict[Action, float],
    values: ValueFunction,
) -> float:
    """Evaluate one Bellman expectation backup for a state."""
    new_value = 0.0
    for action, action_probability in action_distribution.items():
        for probability, next_state, reward, done in env.get_transitions(state, action):
            bootstrap_value = 0.0 if done else values[next_state]
            new_value += action_probability * probability * (
                reward + env.gamma * bootstrap_value
            )
    return new_value


def _expected_action_value(
    env: PlanningGridWorld,
    state: State,
    action: Action,
    values: ValueFunction,
) -> float:
    """Evaluate one Bellman optimality action backup for a state-action pair."""
    action_value = 0.0
    for probability, next_state, reward, done in env.get_transitions(state, action):
        bootstrap_value = 0.0 if done else values[next_state]
        action_value += probability * (reward + env.gamma * bootstrap_value)
    return action_value


def _greedy_action_and_value(
    env: PlanningGridWorld,
    state: State,
    values: ValueFunction,
) -> tuple[Action, float]:
    """Return the first stable greedy action and its Bellman value."""
    actions = env.get_actions(state)
    if not actions:
        raise ValueError(f"Cannot derive a greedy action for terminal state {state}")

    best_action = actions[0]
    best_value = _expected_action_value(env, state, best_action, values)
    for action in actions[1:]:
        action_value = _expected_action_value(env, state, action, values)
        if action_value > best_value:
            best_action = action
            best_value = action_value
    return best_action, best_value


def _greedy_policy(env: PlanningGridWorld, values: ValueFunction) -> Policy:
    """Extract a deterministic greedy policy from a value function."""
    policy: Policy = {}
    for state in env.get_states():
        if not env.is_terminal(state):
            action, _ = _greedy_action_and_value(env, state, values)
            policy[state] = action
    return policy


def _serialize_value_function(values: ValueFunction) -> dict[str, float]:
    """Convert a value function to a JSON-serializable mapping."""
    return {
        _serialize_state(state): float(value)
        for state, value in values.items()
    }


def _serialize_policy(policy: Policy) -> dict[str, Any]:
    """Convert deterministic or stochastic policies to JSON-safe values."""
    serialized: dict[str, Any] = {}
    for state, entry in policy.items():
        if isinstance(entry, str):
            serialized[_serialize_state(state)] = entry
        else:
            serialized[_serialize_state(state)] = {
                action: float(probability)
                for action, probability in entry.items()
            }
    return serialized


def _print_planning_progress_header(algorithm_name: str, verbose: int) -> None:
    """Print planning convergence progress header."""
    print_progress_header(
        f"[{algorithm_name}] Convergence progress",
        ("iter", "residual", "backups"),
        (6, 10, 7),
        verbose,
    )


def _print_planning_progress_row(
    iteration: int,
    residual: float,
    backups: int,
    verbose: int,
) -> None:
    """Print one planning convergence progress row."""
    print_progress_row(
        (iteration, format_scientific(residual), backups),
        (6, 10, 7),
        verbose,
    )


def _validated_log_interval(log_interval: int) -> int:
    """Return a positive logging interval."""
    if log_interval <= 0:
        raise ValueError(f"log_interval must be positive, got {log_interval}")
    return log_interval
