"""Sample-based Grid-world environment for learning algorithms."""

from __future__ import annotations

import random
from typing import Any

from envs.grid_world import Action, GridWorldBase, State, Transition
from envs.planning_grid_world import PlanningGridWorld


class LearningGridWorld(GridWorldBase):
    """Grid-world environment for sample-based reinforcement learning.

    Learning algorithms should train only through transition samples
    (S, A, R, S') obtained from `reset()` and `step(action)`. The environment may
    reuse transition logic internally, but training code should not query the MDP
    model directly.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rng = random.Random(self.seed)
        self._model = PlanningGridWorld(
            grid_size=self.grid_size,
            start_state=self.start_state,
            goal_states=set(self.goal_states),
            trap_states=set(self.trap_states),
            wall_states=set(self.wall_states),
            actions=self.actions,
            reward_config=dict(self.reward_config),
            gamma=self.gamma,
            seed=self.seed,
        )

    def reset(self) -> State:
        """Reset the episode and return the start state."""
        self.current_state = self.start_state
        return self.current_state

    def step(self, action: Action) -> tuple[State, float, bool, dict[str, Any]]:
        """Sample one transition for the given action."""
        transitions = self._model.transition_prob(self.current_state, action)
        probability, next_state, reward, done = self._sample_transition(transitions)
        self.current_state = next_state
        return next_state, reward, done, {"probability": probability}

    def transition_prob(self, state: State, action: Action) -> list[Transition]:
        """Return model transitions for diagnostics, not for learner training.

        TODO: Keep learning agents decoupled from this method. It exists only to
        satisfy the shared environment interface and support tests/analysis.
        """
        return self._model.transition_prob(state, action)

    def _sample_transition(self, transitions: list[Transition]) -> Transition:
        """Sample a transition from a probability distribution."""
        threshold = self._rng.random()
        cumulative = 0.0
        for transition in transitions:
            cumulative += transition[0]
            if threshold <= cumulative:
                return transition
        return transitions[-1]
