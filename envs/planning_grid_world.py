"""Planning Grid-world environment with explicit MDP model access."""

from __future__ import annotations

from typing import Any

from envs.grid_world import Action, GridWorldBase, State, Transition


class PlanningGridWorld(GridWorldBase):
    """Grid-world environment for model-based planning algorithms.

    Planning algorithms are allowed to query the full MDP model, including
    P(s'|s,a) and r(s,a,s'). The current skeleton uses deterministic movement as
    a simple placeholder and can later be extended to stochastic transitions.
    """

    def reset(self) -> State:
        """Reset to the configured start state."""
        self.current_state = self.start_state
        return self.current_state

    def step(self, action: Action) -> tuple[State, float, bool, dict[str, Any]]:
        """Take one deterministic transition using the model helper."""
        transition = self.get_transitions(self.current_state, action)[0]
        _, next_state, reward, done = transition
        self.current_state = next_state
        return next_state, reward, done, {}

    def get_transitions(self, state: State, action: Action) -> list[Transition]:
        """Return `(probability, next_state, reward, done)` transitions.
        """
        if self.is_terminal(state):
            return [(1.0, state, 0.0, True)]

        next_state = self._candidate_next_state(state, action)
        reward = self._reward_for(next_state)
        done = self.is_terminal(next_state)
        return [(1.0, next_state, reward, done)]
