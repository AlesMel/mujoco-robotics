"""Training utilities — callbacks, PPO helpers, shared config."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .callbacks import BestEpisodeVideoCallback
    from .ppo_config import PPOTrainConfig


def __getattr__(name: str):
    if name == "BestEpisodeVideoCallback":
        from .callbacks import BestEpisodeVideoCallback

        return BestEpisodeVideoCallback
    if name == "PPOTrainConfig":
        from .ppo_config import PPOTrainConfig

        return PPOTrainConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BestEpisodeVideoCallback", "PPOTrainConfig"]
