"""Reach task configuration objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from mujoco_robot.envs.reach.mdp import ReachMDPCfg


@dataclass
class ReachTaskConfig:
    """High-level configuration for the reach task."""

    robot: str = "ur3e"
    control_variant: str = "ik_rel"
    time_limit: int | None = None
    seed: int | None = None
    render_mode: str | None = None
    # Optional manager-based MDP override (actions/observations/rewards/terminations).
    mdp_cfg: ReachMDPCfg | None = None
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
