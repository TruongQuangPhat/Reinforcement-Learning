"""Tests for model-based planning algorithms."""

from __future__ import annotations

import unittest

from agents.planning import (
    LinearProgrammingPlanner,
    PolicyEvaluation,
    PolicyIteration,
    ValueIteration,
)
from envs.planning_grid_world import PlanningGridWorld
from utils.metrics import max_abs_error, policy_agreement


class PlanningAlgorithmTests(unittest.TestCase):
    """Numerical and convergence checks for planning algorithms."""

    def test_policy_evaluation_converges_for_fixed_policy(self) -> None:
        env = PlanningGridWorld(
            grid_size=(2, 2),
            start_state=(0, 0),
            goal_states={(0, 1)},
            trap_states=set(),
            wall_states=set(),
            gamma=0.9,
        )
        policy = {(0, 0): "right", (1, 0): "up", (1, 1): "up"}
        evaluator = PolicyEvaluation(env, policy=policy, theta=1e-10)

        values = evaluator.run()
        metrics = evaluator.get_metrics()

        self.assertEqual(metrics["status"], "converged")
        self.assertLess(metrics["final_bellman_residual"], 1e-10)
        self.assertAlmostEqual(values[(0, 0)], 10.0)
        self.assertEqual(values[(0, 1)], 0.0)

    def test_policy_evaluation_rejects_invalid_policy_distribution(self) -> None:
        env = PlanningGridWorld(
            grid_size=(2, 2),
            goal_states={(1, 1)},
            trap_states=set(),
            wall_states=set(),
        )
        policy = {(0, 0): {"right": 0.75, "down": 0.75}}
        evaluator = PolicyEvaluation(env, policy=policy)

        with self.assertRaises(ValueError):
            evaluator.run()

    def test_value_iteration_converges_and_extracts_greedy_policy(self) -> None:
        env = PlanningGridWorld(
            grid_size=(2, 2),
            start_state=(0, 0),
            goal_states={(0, 1)},
            trap_states=set(),
            wall_states=set(),
            gamma=0.9,
        )
        planner = ValueIteration(env, theta=1e-10)

        values, policy = planner.run()
        metrics = planner.get_metrics()

        self.assertEqual(metrics["status"], "converged")
        self.assertLess(metrics["final_bellman_residual"], 1e-10)
        self.assertEqual(policy[(0, 0)], "right")
        self.assertAlmostEqual(values[(0, 0)], 10.0)

    def test_policy_iteration_matches_value_iteration(self) -> None:
        env = PlanningGridWorld(
            grid_size=(4, 4),
            start_state=(0, 0),
            goal_states={(3, 3)},
            trap_states={(1, 1)},
            wall_states={(2, 1)},
            gamma=0.9,
        )
        value_iteration = ValueIteration(env, theta=1e-8)
        vi_values, vi_policy = value_iteration.run()
        policy_iteration = PolicyIteration(env, theta=1e-8)
        pi_values, pi_policy = policy_iteration.run()
        states = [state for state in env.get_states() if not env.is_terminal(state)]

        self.assertEqual(policy_iteration.get_metrics()["status"], "converged")
        self.assertLess(max_abs_error(pi_values, vi_values), 1e-6)
        self.assertEqual(policy_agreement(pi_policy, vi_policy, states), 1.0)

    def test_policy_iteration_does_not_report_false_convergence(self) -> None:
        env = PlanningGridWorld(
            grid_size=(2, 2),
            start_state=(0, 0),
            goal_states={(1, 1)},
            trap_states=set(),
            wall_states=set(),
            reward_config={"step": 1.0, "goal": 0.0},
            gamma=0.99,
        )
        planner = PolicyIteration(env, theta=1e-12, max_iterations=1)

        planner.run()
        metrics = planner.get_metrics()

        self.assertEqual(metrics["status"], "max_iterations_reached")
        self.assertTrue(metrics["policy_stable"])
        self.assertFalse(metrics["final_policy_evaluation_converged"])

    def test_linear_programming_matches_value_iteration(self) -> None:
        env = PlanningGridWorld(
            grid_size=(4, 4),
            start_state=(0, 0),
            goal_states={(3, 3)},
            trap_states={(1, 1)},
            wall_states={(2, 1)},
            gamma=0.9,
        )
        value_iteration = ValueIteration(env, theta=1e-8)
        vi_values, vi_policy = value_iteration.run()
        lp_planner = LinearProgrammingPlanner(env)
        lp_values, lp_policy = lp_planner.run()
        states = [state for state in env.get_states() if not env.is_terminal(state)]

        self.assertEqual(lp_planner.get_metrics()["status"], "optimal")
        self.assertEqual(lp_planner.get_metrics()["number_of_variables"], env.num_states())
        self.assertLess(max_abs_error(lp_values, vi_values), 1e-6)
        self.assertEqual(policy_agreement(lp_policy, vi_policy, states), 1.0)


if __name__ == "__main__":
    unittest.main()
