"""Gymnasium-facing base class for manager-based RL envs."""
from __future__ import annotations

from abc import ABC
from typing import Any

import gymnasium

from .base_env import ManagerBasedEnv


class ManagerBasedRLEnv(gymnasium.Env, ManagerBasedEnv, ABC):
    """Abstract RL environment composed from manager terms and cfg."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: Any) -> None:
        gymnasium.Env.__init__(self)
        ManagerBasedEnv.__init__(self, cfg=cfg)

