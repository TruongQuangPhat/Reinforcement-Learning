"""Tests for shared Grid-world environment behavior."""

from __future__ import annotations

import unittest

from envs.learning_grid_world import LearningGridWorld
from envs.planning_grid_world import PlanningGridWorld


class GridWorldEnvironmentTests(unittest.TestCase):
    """Validate environment interfaces used by planning and learning agents."""

    def test_reset_returns_configured_start_state(self) -> None:
        env = LearningGridWorld(grid_size=(3, 3), start_state=(1, 1))

        self.assertEqual(env.reset(), (1, 1))
        self.assertEqual(env.current_state, (1, 1))

    def test_step_respects_boundaries_and_walls(self) -> None:
        env = LearningGridWorld(
            grid_size=(3, 3),
            start_state=(0, 0),
            goal_states={(2, 2)},
            trap_states=set(),
            wall_states={(0, 1)},
        )

        next_state, reward, done, _ = env.step("up")
        self.assertEqual(next_state, (0, 0))
        self.assertEqual(reward, env.reward_config["step"])
        self.assertFalse(done)

        env.reset()
        next_state, reward, done, _ = env.step("right")
        self.assertEqual(next_state, (0, 0))
        self.assertEqual(reward, env.reward_config["step"])
        self.assertFalse(done)

    def test_goal_and_trap_rewards_end_episode(self) -> None:
        goal_env = LearningGridWorld(
            grid_size=(2, 2),
            start_state=(0, 0),
            goal_states={(0, 1)},
            trap_states=set(),
            wall_states=set(),
            reward_config={"goal": 7.0},
        )
        next_state, reward, done, _ = goal_env.step("right")

        self.assertEqual(next_state, (0, 1))
        self.assertEqual(reward, 7.0)
        self.assertTrue(done)

        trap_env = LearningGridWorld(
            grid_size=(2, 2),
            start_state=(0, 0),
            goal_states={(1, 1)},
            trap_states={(0, 1)},
            wall_states=set(),
            reward_config={"trap": -9.0},
        )
        next_state, reward, done, _ = trap_env.step("right")

        self.assertEqual(next_state, (0, 1))
        self.assertEqual(reward, -9.0)
        self.assertTrue(done)

    def test_step_from_terminal_state_returns_zero_reward_done(self) -> None:
        env = LearningGridWorld(
            grid_size=(2, 2),
            start_state=(0, 0),
            goal_states={(0, 0)},
            trap_states=set(),
            wall_states=set(),
        )

        next_state, reward, done, _ = env.step("right")

        self.assertEqual(next_state, (0, 0))
        self.assertEqual(reward, 0.0)
        self.assertTrue(done)

    def test_planning_transition_is_valid_probability_distribution(self) -> None:
        env = PlanningGridWorld(
            grid_size=(3, 3),
            start_state=(0, 0),
            goal_states={(2, 2)},
            trap_states=set(),
            wall_states={(1, 1)},
        )

        transitions = env.get_transitions((0, 0), "right")

        self.assertEqual(len(transitions), 1)
        self.assertEqual(sum(probability for probability, *_ in transitions), 1.0)
        probability, next_state, reward, done = transitions[0]
        self.assertEqual(probability, 1.0)
        self.assertEqual(next_state, (0, 1))
        self.assertEqual(reward, env.reward_config["step"])
        self.assertFalse(done)

    def test_terminal_planning_transition_is_absorbing(self) -> None:
        env = PlanningGridWorld(
            grid_size=(2, 2),
            start_state=(0, 0),
            goal_states={(0, 1)},
            trap_states=set(),
            wall_states=set(),
        )

        self.assertEqual(env.get_transitions((0, 1), "left"), [(1.0, (0, 1), 0.0, True)])

    def test_invalid_overlapping_special_states_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LearningGridWorld(
                grid_size=(2, 2),
                goal_states={(1, 1)},
                trap_states={(1, 1)},
                wall_states=set(),
            )


if __name__ == "__main__":
    unittest.main()
