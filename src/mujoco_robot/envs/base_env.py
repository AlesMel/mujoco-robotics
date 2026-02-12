"""Base protocol for manager-based task environments."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ManagerBasedEnv(ABC):
    """Abstract base for manager-composed task environments."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    @abstractmethod
    def reset(self, *args, **kwargs):
        """Reset the environment and return initial observation/state."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """Advance the environment by one control step."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release resources associated with the environment."""
        raise NotImplementedError

