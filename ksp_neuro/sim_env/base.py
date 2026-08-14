"""Base class for KSP environments (simulator or live game)."""

from __future__ import annotations

import abc
from typing import Any, Dict, Tuple


class BaseEnvironment(abc.ABC):
    """Abstract environment interface for neuroevolution training."""

    @abc.abstractmethod
    def reset(self) -> "np.ndarray":  # noqa: F821 — numpy added at runtime
        """Reset to initial state; return observation vector."""

    @abc.abstractmethod
    def step(
        self, action: "np.ndarray"
    ) -> Tuple["np.ndarray", float, bool, Dict[str, Any]]:  # noqa: F821
        """Execute *action*, return (obs, reward, done, info)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Clean up resources."""

    @property
    @abc.abstractmethod
    def observation_space(self):
        pass

    @property
    @abc.abstractmethod
    def action_space(self):
        pass


__all__ = ["BaseEnvironment"]
