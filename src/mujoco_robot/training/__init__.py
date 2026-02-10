"""Training utilities — callbacks, PPO helpers."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .callbacks import BestEpisodeVideoCallback


def __getattr__(name: str):
    if name == "BestEpisodeVideoCallback":
        from .callbacks import BestEpisodeVideoCallback

        return BestEpisodeVideoCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BestEpisodeVideoCallback"]
