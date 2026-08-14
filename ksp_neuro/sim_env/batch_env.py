"""Batch environment wrapper — bridges vectorized sim and NE engine.

This module provides a high-level interface for running many parallel episodes
through the vectorized simulator, which is the recommended way to evaluate
the neuroevolution population during training.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional


class BatchEnv:
    """Manages N parallel simulation instances for efficient NE evaluation.

    Handles episode management across many agents — resetting completed episodes,
    running action policies, and collecting per-episode metrics.

    Parameters
    ----------
    n_agents : int
        Number of parallel simulation instances.
    max_steps_per_episode : int
        Maximum steps before forced reset (default 3600 = ~6 min).
    dt : float
        Physics timestep in seconds.
    seed : int | None
        Base random seed for reproducibility.

    Usage:
        >>> batch = BatchEnv(n_agents=200, max_steps_per_episode=3600)
        >>> obs = batch.reset()          # shape (n_agents, 16)
        >>> episode_rewards = np.zeros(n_agents)
        >>> for step in range(max_steps):
        ...     actions = policy(obs)    # neural net forward pass
        ...     obs, rewards, dones, info = batch.step(actions)
        ...     episode_rewards += rewards * ~dones  # accumulate only active
        ...     if np.all(dones): break
        >>> final_scores = episode_rewards + rewards * dones  # add last step
    """

    def __init__(
        self,
        n_agents: int = 200,
        max_steps_per_episode: int = 3600,
        dt: float = 0.1,
        seed: Optional[int] = None,
    ):
        from ksp_neuro.sim_env.vectorized_sim import VectorizedOrbitSimulator

        self.n_agents = n_agents
        self.max_steps_per_episode = max_steps_per_episode
        self._sim = VectorizedOrbitSimulator(
            n_agents=n_agents, dt=dt, max_steps=max_steps_per_episode, seed=seed
        )
        self._active = np.ones(n_agents, dtype=bool)  # which agents are still running
        self._episode_rewards = np.zeros(n_agents, dtype=np.float64)

    @property
    def observation_space(self):
        return len(_OBS_KEYS), self.n_agents

    @property
    def action_space(self):
        return len(_ACTION_KEYS), "float32"

    def reset(
        self,
        seed_offset: int = 0,
    ) -> np.ndarray:
        """Reset all agents and return initial observations.

        Parameters
        ----------
        seed_offset : int
            Offset for per-agent random seeds (for varied initial conditions).

        Returns
        -------
        obs : np.ndarray  shape (n_agents, 16)
        """
        self._active.fill(True)
        self._episode_rewards.fill(0.0)
        return self._sim.reset(seed_offset=seed_offset)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Execute one tick for all agents.

        Parameters
        ----------
        actions : np.ndarray  shape (n_agents, 8)
            Control vector keyed by ``_ACTION_KEYS``.

        Returns
        -------
        obs : np.ndarray  shape (n_agents, 16) — observations for active agents; zeros for done
        rewards : np.ndarray  shape (n_agents,) — rewards (zeroed for already-done agents)
        dones : np.ndarray  shape (n_agents,), bool — which agents have terminated
        info : dict with per-agent tracking arrays
        """
        # Zero out actions for done agents to prevent further simulation
        safe_actions = actions.copy()
        safe_actions[~self._active] = 0.0

        obs, rewards, dones_raw, info = self._sim.step(safe_actions)

        # Only update active agents
        new_active_mask = ~dones_raw & self._active
        self._active &= new_active_mask

        # Zero out observations and rewards for done agents
        obs[~self._active] = 0.0
        rewards[~self._active] = 0.0

        return obs, rewards, dones_raw, info

    def close(self) -> None:
        """Clean up simulation resources."""
        self._sim.close()


# Re-export observation/action keys for compatibility
from ksp_neuro.sim_env.vectorized_sim import _OBS_KEYS, _ACTION_KEYS  # noqa: E402

__all__ = ["BatchEnv", "_OBS_KEYS", "_ACTION_KEYS"]
