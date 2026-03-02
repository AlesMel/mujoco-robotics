"""Training utilities — callbacks, PPO config, helpers."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .callbacks import BestEpisodeVideoCallback
    from .ppo_cfg import NormalizeCfg, PPOCfg, TrainCfg, build_sb3_ppo


def __getattr__(name: str):
    if name == "BestEpisodeVideoCallback":
        from .callbacks import BestEpisodeVideoCallback

        return BestEpisodeVideoCallback
    if name in ("PPOCfg", "NormalizeCfg", "TrainCfg", "build_sb3_ppo"):
        from . import ppo_cfg

        return getattr(ppo_cfg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BestEpisodeVideoCallback",
    "NormalizeCfg",
    "PPOCfg",
    "TrainCfg",
    "build_sb3_ppo",
]
