"""Shared Grid-world environment abstraction.

This module defines the base interface for an 8x8 Grid-world navigation
environment. Concrete environments may expose a full MDP model for planning or
only sample-based interaction for learning, but they share the same state,
action, reward, and terminal-state concepts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeAlias

State: TypeAlias = tuple[int, int]
Action: TypeAlias = str
Transition: TypeAlias = tuple[float, State, float, bool]


class GridWorldBase(ABC):
    """Base interface for finite Grid-world MDP environments.

    The environment represents a finite Markov Decision Process
    M = (S, A, P, r, gamma). Subclasses decide whether the full transition model
    is public (`PlanningGridWorld`) or hidden behind `reset()` and `step()`
    (`LearningGridWorld`).
    """

    def __init__(
        self,
        grid_size: tuple[int, int] = (8, 8),
        start_state: State = (0, 0),
        goal_states: set[State] | None = None,
        trap_states: set[State] | None = None,
        wall_states: set[State] | None = None,
        actions: tuple[Action, ...] = ("up", "down", "left", "right"),
        reward_config: dict[str, float] | None = None,
        gamma: float = 0.99,
        seed: int | None = 42,
    ) -> None:
        self.grid_size = grid_size
        self.start_state = start_state
        self.goal_states = goal_states or {(7, 7)}
        self.trap_states = trap_states or {(3, 3), (4, 4)}
        self.wall_states = wall_states or {(2, 2), (2, 3), (5, 5)}
        self.actions = actions
        self.reward_config = reward_config or {
            "step": -1.0,
            "goal": 10.0,
            "trap": -10.0,
            "wall": -1.0,
        }
        self.gamma = gamma
        self.seed = seed
        self.current_state = start_state

    def get_states(self) -> list[State]:
        """Return all valid non-wall states in the grid."""
        rows, cols = self.grid_size
        return [
            (row, col)
            for row in range(rows)
            for col in range(cols)
            if not self.is_wall((row, col))
        ]

    def get_actions(self, state: State | None = None) -> tuple[Action, ...]:
        """Return available actions for a state.

        Terminal and wall states have no meaningful outgoing actions.
        """
        if state is not None and (self.is_terminal(state) or self.is_wall(state)):
            return ()
        return self.actions

    def is_terminal(self, state: State) -> bool:
        """Return whether a state ends an episode."""
        return state in self.goal_states or state in self.trap_states

    def is_wall(self, state: State) -> bool:
        """Return whether a state is blocked by a wall."""
        return state in self.wall_states

    def in_bounds(self, state: State) -> bool:
        """Return whether a state coordinate is inside the grid."""
        row, col = state
        rows, cols = self.grid_size
        return 0 <= row < rows and 0 <= col < cols

    def render(self) -> None:
        """Print a simple text view of the current grid state."""
        # TODO: Replace with richer rendering or matplotlib visualization later.
        rows, cols = self.grid_size
        for row in range(rows):
            cells: list[str] = []
            for col in range(cols):
                state = (row, col)
                if state == self.current_state:
                    cells.append("A")
                elif state == self.start_state:
                    cells.append("S")
                elif state in self.goal_states:
                    cells.append("G")
                elif state in self.trap_states:
                    cells.append("T")
                elif state in self.wall_states:
                    cells.append("#")
                else:
                    cells.append(".")
            print(" ".join(cells))

    @abstractmethod
    def reset(self) -> State:
        """Reset the environment and return the initial state."""

    @abstractmethod
    def step(self, action: Action) -> tuple[State, float, bool, dict[str, Any]]:
        """Apply an action and return `(next_state, reward, done, info)`."""

    @abstractmethod
    def transition_prob(self, state: State, action: Action) -> list[Transition]:
        """Return model transitions `(probability, next_state, reward, done)`."""
