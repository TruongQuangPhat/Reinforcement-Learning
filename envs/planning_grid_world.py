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

    _ACTION_DELTAS: dict[Action, State] = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    def reset(self) -> State:
        """Reset to the configured start state."""
        self.current_state = self.start_state
        return self.current_state

    def step(self, action: Action) -> tuple[State, float, bool, dict[str, Any]]:
        """Take one deterministic transition using the model helper."""
        transition = self.transition_prob(self.current_state, action)[0]
        _, next_state, reward, done = transition
        self.current_state = next_state
        return next_state, reward, done, {}

    def transition_prob(self, state: State, action: Action) -> list[Transition]:
        """Return `(probability, next_state, reward, done)` transitions.

        TODO: Extend this method if the assignment requires stochastic movement
        such as slip probability or action noise.
        """
        if self.is_terminal(state):
            return [(1.0, state, 0.0, True)]

        next_state = self._candidate_next_state(state, action)
        reward = self._reward_for(next_state)
        done = self.is_terminal(next_state)
        return [(1.0, next_state, reward, done)]

    def _candidate_next_state(self, state: State, action: Action) -> State:
        """Compute the deterministic next state for a candidate action."""
        row_delta, col_delta = self._ACTION_DELTAS.get(action, (0, 0))
        candidate = (state[0] + row_delta, state[1] + col_delta)
        if not self.in_bounds(candidate) or self.is_wall(candidate):
            return state
        return candidate

    def _reward_for(self, state: State) -> float:
        """Return immediate reward for entering a state."""
        if state in self.goal_states:
            return self.reward_config["goal"]
        if state in self.trap_states:
            return self.reward_config["trap"]
        return self.reward_config["step"]
