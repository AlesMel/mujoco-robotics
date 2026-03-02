"""Training utilities — callbacks, PPO helpers."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .callbacks import BestEpisodeVideoCallback, RunDirCheckpointCallback


def __getattr__(name: str):
    if name == "BestEpisodeVideoCallback":
        from .callbacks import BestEpisodeVideoCallback

        return BestEpisodeVideoCallback
    if name == "RunDirCheckpointCallback":
        from .callbacks import RunDirCheckpointCallback

        return RunDirCheckpointCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BestEpisodeVideoCallback", "RunDirCheckpointCallback"]
