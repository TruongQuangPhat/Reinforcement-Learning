"""Tests for Grid-world visualization helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from envs.planning_grid_world import PlanningGridWorld
from utils.visualization import (
    plot_bellman_residual,
    plot_comparison_bar,
    plot_grid_world_layout,
    plot_policy_arrows,
    plot_policy_changes,
    plot_value_heatmap,
)


class VisualizationTests(unittest.TestCase):
    """Validate visualization helpers without requiring an interactive display."""

    def test_plot_grid_world_layout_returns_figure_and_axis(self) -> None:
        env = PlanningGridWorld()

        figure, axis = plot_grid_world_layout(env, show=False)

        self.assertEqual(type(figure).__name__, "Figure")
        self.assertEqual(type(axis).__name__, "Axes")

    def test_plot_grid_world_layout_saves_file(self) -> None:
        env = PlanningGridWorld()

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "grid_layout.png"
            plot_grid_world_layout(env, save_path=save_path, show=False)

            self.assertTrue(save_path.exists())
            self.assertGreater(save_path.stat().st_size, 0)

    def test_plot_value_heatmap_returns_figure_and_axis(self) -> None:
        env = PlanningGridWorld()
        values = {state: 0.0 for state in env.get_states()}

        figure, axis = plot_value_heatmap(values, env, "Values", show=False)

        self.assertEqual(type(figure).__name__, "Figure")
        self.assertEqual(type(axis).__name__, "Axes")

    def test_plot_policy_arrows_returns_figure_and_axis(self) -> None:
        env = PlanningGridWorld()
        policy = {
            state: "right"
            for state in env.get_states()
            if not env.is_terminal(state)
        }

        figure, axis = plot_policy_arrows(policy, env, "Policy", show=False)

        self.assertEqual(type(figure).__name__, "Figure")
        self.assertEqual(type(axis).__name__, "Axes")

    def test_plot_bellman_residual_returns_figure_and_axis(self) -> None:
        figure, axis = plot_bellman_residual([1.0, 0.5, 0.1], "Residual", show=False)

        self.assertEqual(type(figure).__name__, "Figure")
        self.assertEqual(type(axis).__name__, "Axes")
        self.assertEqual(axis.get_ylabel(), "Bellman residual")

    def test_plot_policy_changes_returns_figure_and_axis(self) -> None:
        figure, axis = plot_policy_changes([10, 5, 2, 0], show=False)

        self.assertEqual(type(figure).__name__, "Figure")
        self.assertEqual(type(axis).__name__, "Axes")
        self.assertEqual(axis.get_xlabel(), "Policy improvement iteration")
        self.assertEqual(axis.get_ylabel(), "Number of changed states")
        self.assertNotIn("Bellman residual", axis.get_ylabel())

    def test_plot_policy_changes_saves_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "policy_changes.png"
            plot_policy_changes([10, 5, 2, 0], save_path=save_path, show=False)

            self.assertTrue(save_path.exists())
            self.assertGreater(save_path.stat().st_size, 0)

    def test_plot_comparison_bar_returns_figure_and_axis(self) -> None:
        figure, axis = plot_comparison_bar({"A": 1.0, "B": 2.0}, "Comparison", show=False)

        self.assertEqual(type(figure).__name__, "Figure")
        self.assertEqual(type(axis).__name__, "Axes")


if __name__ == "__main__":
    unittest.main()
