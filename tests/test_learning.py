"""Sanity tests for model-free learning algorithms."""

from __future__ import annotations

import unittest

from agents.learning import QLearning, SARSA, TDLambda, TDZero
from envs.learning_grid_world import LearningGridWorld


class GuardedLearningGridWorld(LearningGridWorld):
    """Learning environment that fails if learners query the model directly."""

    def transition_prob(self, state, action):  # type: ignore[no-untyped-def]
        raise AssertionError("Learning agents must not call transition_prob().")


class LearningAlgorithmTests(unittest.TestCase):
    """Focused correctness checks for the learning module."""

    def test_q_table_shape_matches_non_terminal_state_actions(self) -> None:
        env = LearningGridWorld(seed=1)
        learner = QLearning(env, episodes=1, seed=1)

        expected_size = sum(len(env.get_actions(state)) for state in env.get_states())

        self.assertEqual(len(learner.get_q_table()), expected_size)

    def test_q_learning_updates_at_least_one_state_action(self) -> None:
        env = GuardedLearningGridWorld(seed=2)
        learner = QLearning(env, episodes=5, max_steps_per_episode=20, seed=2)

        learner.train()

        self.assertTrue(any(value != 0.0 for value in learner.get_q_table().values()))
        self.assertGreater(learner.get_metrics()["q_updates"], 0)

    def test_sarsa_records_expected_training_metrics(self) -> None:
        env = GuardedLearningGridWorld(seed=3)
        learner = SARSA(env, episodes=5, max_steps_per_episode=20, seed=3)

        learner.train()
        metrics = learner.get_metrics()

        for key in (
            "episode_returns",
            "moving_average_returns",
            "episode_steps",
            "success_rate",
            "trap_rate",
            "td_errors",
            "learned_policy",
            "environment_steps",
        ):
            self.assertIn(key, metrics)

    def test_td_zero_one_step_updates_only_start_state(self) -> None:
        env = GuardedLearningGridWorld(seed=4)
        learner = TDZero(env, episodes=1, max_steps_per_episode=1, seed=4)

        learner.train()
        values = learner.get_value_function()
        changed_states = [state for state, value in values.items() if value != 0.0]

        self.assertEqual(changed_states, [env.start_state])

    def test_td_lambda_updates_eligible_previous_states(self) -> None:
        env = GuardedLearningGridWorld(seed=5)
        policy = {
            (0, 0): "right",
            (0, 1): "right",
        }
        learner = TDLambda(
            env,
            episodes=1,
            max_steps_per_episode=2,
            policy=policy,
            seed=5,
        )

        learner.train()
        values = learner.get_value_function()

        self.assertNotEqual(values[(0, 0)], 0.0)
        self.assertNotEqual(values[(0, 1)], 0.0)

    def test_same_seed_gives_same_q_learning_values(self) -> None:
        env_a = GuardedLearningGridWorld(seed=6)
        env_b = GuardedLearningGridWorld(seed=6)
        learner_a = QLearning(env_a, episodes=10, max_steps_per_episode=25, seed=6)
        learner_b = QLearning(env_b, episodes=10, max_steps_per_episode=25, seed=6)

        learner_a.train()
        learner_b.train()

        self.assertEqual(learner_a.get_q_table(), learner_b.get_q_table())


if __name__ == "__main__":
    unittest.main()
