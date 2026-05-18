"""Sample-based Grid-world environment for learning algorithms."""

from __future__ import annotations

from typing import Any

from envs.grid_world import Action, GridWorldBase, State


class LearningGridWorld(GridWorldBase):
    """Grid-world environment for sample-based reinforcement learning.

    Learning algorithms should train only through transition samples
    (S, A, R, S') obtained from `reset()` and `step(action)`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def reset(self) -> State:
        """Reset the episode and return the start state."""
        self.current_state = self.start_state
        return self.current_state

    def step(self, action: Action) -> tuple[State, float, bool, dict[str, Any]]:
        """Apply one deterministic environment interaction."""
        if self.is_terminal(self.current_state):
            return self.current_state, 0.0, True, {}

        next_state = self._candidate_next_state(self.current_state, action)
        reward = self._reward_for(next_state)
        done = self.is_terminal(next_state)
        self.current_state = next_state
        return next_state, reward, done, {}
