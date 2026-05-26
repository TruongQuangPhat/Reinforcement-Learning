"""Integration tests for experiment runners, logs, and generated figures."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts import run_experiments


class ExperimentRunnerTests(unittest.TestCase):
    """Validate experiment entry points without writing to the repository logs."""

    def test_experiment_runners_write_valid_logs_and_figures(self) -> None:
        old_cwd = Path.cwd()
        old_episodes = run_experiments.LEARNING_EPISODES
        old_max_steps = run_experiments.LEARNING_MAX_STEPS
        old_planning_dir = run_experiments.PLANNING_FIGURE_DIR
        old_learning_dir = run_experiments.LEARNING_FIGURE_DIR
        old_comparison_dir = run_experiments.COMPARISON_FIGURE_DIR

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.chdir(temp_dir)
                run_experiments.LEARNING_EPISODES = 5
                run_experiments.LEARNING_MAX_STEPS = 20
                run_experiments.PLANNING_FIGURE_DIR = Path("report/figures/planning")
                run_experiments.LEARNING_FIGURE_DIR = Path("report/figures/learning")
                run_experiments.COMPARISON_FIGURE_DIR = Path("report/figures/comparison")

                planning_summary = run_experiments.run_planning_experiments()
                learning_summary = run_experiments.run_learning_experiments()
                comparison_summary = run_experiments.run_comparison_experiments()

                self.assertEqual(planning_summary["status"], "completed")
                self.assertEqual(learning_summary["status"], "completed")
                self.assertEqual(comparison_summary["status"], "completed")

                self._assert_log_group_is_valid("planning")
                self._assert_log_group_is_valid("learning")
                self._assert_paths_exist(planning_summary["figure_paths"])
                self._assert_paths_exist(learning_summary["comparison_figure_paths"])
                self._assert_paths_exist(comparison_summary["figure_paths"])

                learning_metrics = self._load_json("logs/learning/training_metrics.json")
                self.assertIn("mse_vs_policy_evaluation", learning_metrics["td_zero"])
                self.assertIn(
                    "mse_vs_policy_evaluation_checkpoints",
                    learning_metrics["td_zero"],
                )
                self.assertIn("mse_vs_policy_evaluation", learning_metrics["td_lambda"])
                self.assertIn("mse_vs_value_iteration", learning_metrics["sarsa"])
                self.assertIn("window_success_rates", learning_metrics["sarsa"])
                self.assertIn("mse_vs_value_iteration", learning_metrics["q_learning"])
                self.assertIn(
                    "policy_agreement_vs_value_iteration",
                    learning_metrics["sarsa"],
                )
                self.assertIn(
                    "policy_agreement_vs_value_iteration",
                    learning_metrics["q_learning"],
                )
                planning_metrics = self._load_json("logs/planning/training_metrics.json")
                self.assertIn("planning_comparison", planning_metrics)
                self.assertIn(
                    "value_error_pi_vs_vi_inf_norm",
                    planning_metrics["planning_comparison"],
                )
                self.assertIn(
                    "value_mse_lp_vs_vi",
                    planning_metrics["planning_comparison"],
                )
            finally:
                os.chdir(old_cwd)
                run_experiments.LEARNING_EPISODES = old_episodes
                run_experiments.LEARNING_MAX_STEPS = old_max_steps
                run_experiments.PLANNING_FIGURE_DIR = old_planning_dir
                run_experiments.LEARNING_FIGURE_DIR = old_learning_dir
                run_experiments.COMPARISON_FIGURE_DIR = old_comparison_dir

    def _assert_log_group_is_valid(self, group: str) -> None:
        for filename in (
            "training_metrics.json",
            "system_metrics.json",
            "experiment_summary.json",
        ):
            path = Path("logs") / group / filename
            self.assertTrue(path.exists(), f"missing {path}")
            self._load_json(path)

    def _assert_paths_exist(self, paths: list[str]) -> None:
        self.assertTrue(paths)
        for path in paths:
            self.assertTrue(Path(path).exists(), f"missing {path}")
            self.assertGreater(Path(path).stat().st_size, 0)

    def _load_json(self, path: str | Path) -> object:
        with Path(path).open("r", encoding="utf-8") as file:
            return json.load(file)


if __name__ == "__main__":
    unittest.main()
