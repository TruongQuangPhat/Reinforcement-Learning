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
    """Base interface for finite Grid-world MDP environments."""

    VALID_ACTIONS: set[Action] = {"up", "down", "left", "right"}
    DEFAULT_REWARD_CONFIG: dict[str, float] = {
        "step": -1.0,
        "goal": 10.0,
        "trap": -10.0,
        "wall": -1.0,
    }
    _ACTION_DELTAS: dict[Action, State] = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }
    _DEFAULT_TRAP_CANDIDATES: tuple[State, ...] = ((3, 3), (4, 4))
    _DEFAULT_WALL_CANDIDATES: tuple[State, ...] = ((2, 2), (2, 3), (5, 5))

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
        self._validate_grid_size()
        self.start_state = start_state
        self.goal_states = self._default_goal_states() if goal_states is None else set(goal_states)
        self.trap_states = self._default_trap_states() if trap_states is None else set(trap_states)
        self.wall_states = self._default_wall_states() if wall_states is None else set(wall_states)
        self.actions = actions
        self.reward_config = dict(self.DEFAULT_REWARD_CONFIG)
        if reward_config is not None:
            self.reward_config.update(reward_config)
        self.gamma = gamma
        self.seed = seed
        self.current_state = start_state

        self._validate_config()
        self.states = self._build_states()
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.idx_to_state = {idx: state for state, idx in self.state_to_idx.items()}

    def _validate_grid_size(self) -> None:
        """Validate grid dimensions before deriving default states."""
        try:
            rows, cols = self.grid_size
        except (TypeError, ValueError) as exc:
            raise ValueError(f"grid_size must be a pair of positive integers, got {self.grid_size}") from exc

        if not isinstance(rows, int) or not isinstance(cols, int):
            raise ValueError(f"grid_size must contain integers, got {self.grid_size}")

        if rows <= 0 or cols <= 0:
            raise ValueError(f"grid_size must contain positive dimensions, got {self.grid_size}")

    def _default_goal_states(self) -> set[State]:
        """Return the default goal at the bottom-right grid cell."""
        rows, cols = self.grid_size
        return {(rows - 1, cols - 1)}

    def _default_trap_states(self) -> set[State]:
        """Return default traps that fit the current grid without overlapping goals."""
        blocked = self.goal_states | {self.start_state}
        return {
            state
            for state in self._DEFAULT_TRAP_CANDIDATES
            if self.in_bounds(state) and state not in blocked
        }

    def _default_wall_states(self) -> set[State]:
        """Return default walls that fit the current grid without overlapping key states."""
        blocked = self.goal_states | self.trap_states | {self.start_state}
        return {
            state
            for state in self._DEFAULT_WALL_CANDIDATES
            if self.in_bounds(state) and state not in blocked
        }

    def _validate_config(self) -> None:
        """Validate Grid-world configuration before experiments run."""
        if not self.in_bounds(self.start_state):
            raise ValueError(f"start_state {self.start_state} is outside grid {self.grid_size}")

        if self.start_state in self.wall_states:
            raise ValueError(f"start_state {self.start_state} cannot be a wall state")

        for name, states in (
            ("goal_states", self.goal_states),
            ("trap_states", self.trap_states),
            ("wall_states", self.wall_states),
        ):
            invalid = [state for state in states if not self.in_bounds(state)]
            if invalid:
                raise ValueError(f"{name} contains out-of-bounds states: {invalid}")

        if self.goal_states & self.wall_states:
            raise ValueError(f"goal_states overlap wall_states: {self.goal_states & self.wall_states}")
        if self.trap_states & self.wall_states:
            raise ValueError(f"trap_states overlap wall_states: {self.trap_states & self.wall_states}")
        if self.goal_states & self.trap_states:
            raise ValueError(f"goal_states overlap trap_states: {self.goal_states & self.trap_states}")

        if not 0.0 <= self.gamma < 1.0:
            raise ValueError(f"gamma must satisfy 0 <= gamma < 1, got {self.gamma}")

        if not self.actions:
            raise ValueError("actions must be non-empty")

        invalid_actions = [action for action in self.actions if action not in self.VALID_ACTIONS]
        if invalid_actions:
            raise ValueError(f"actions contain invalid values: {invalid_actions}")

        for key in self.DEFAULT_REWARD_CONFIG:
            reward = self.reward_config.get(key)
            if not isinstance(reward, (int, float)):
                raise ValueError(f"reward_config[{key!r}] must be numeric, got {reward!r}")

    def _build_states(self) -> list[State]:
        """Build all valid non-wall states once in stable row-major order."""
        rows, cols = self.grid_size
        return [
            (row, col)
            for row in range(rows)
            for col in range(cols)
            if (row, col) not in self.wall_states
        ]

    def get_states(self) -> list[State]:
        """Return a copy of all valid non-wall states."""
        return list(self.states)

    def get_actions(self, state: State | None = None) -> tuple[Action, ...]:
        """Return available actions for a non-terminal, non-wall state."""
        if state is not None and (self.is_terminal(state) or self.is_wall(state)):
            return ()
        return self.actions

    def _candidate_next_state(self, state: State, action: Action) -> State:
        """Compute the deterministic next state for a candidate action."""
        self._validate_action(action)
        row_delta, col_delta = self._ACTION_DELTAS[action]
        candidate = (state[0] + row_delta, state[1] + col_delta)
        if not self.in_bounds(candidate) or self.is_wall(candidate):
            return state
        return candidate

    def _validate_action(self, action: Action) -> None:
        """Raise when an action is not part of this environment's action set."""
        if action not in self.actions:
            raise ValueError(f"Unknown action {action!r}; expected one of {self.actions}")

    def _reward_for(self, state: State) -> float:
        """Return immediate reward for entering a state."""
        if state in self.goal_states:
            return self.reward_config["goal"]
        if state in self.trap_states:
            return self.reward_config["trap"]
        return self.reward_config["step"]

    def state_to_index(self, state: State) -> int:
        """Return the cached integer index for a state."""
        if state not in self.state_to_idx:
            raise ValueError(f"Unknown state: {state}")
        return self.state_to_idx[state]

    def index_to_state(self, index: int) -> State:
        """Return the state for a cached integer index."""
        if index not in self.idx_to_state:
            raise ValueError(f"Unknown state index: {index}")
        return self.idx_to_state[index]

    def num_states(self) -> int:
        """Return the number of valid non-wall states."""
        return len(self.states)

    def num_actions(self) -> int:
        """Return the number of configured actions."""
        return len(self.actions)

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
