"""Reach task entrypoints and config."""

from mujoco_robot.tasks.reach.config import ReachTaskConfig
from mujoco_robot.tasks.reach.factory import make_reach_env, make_reach_gymnasium
from mujoco_robot.envs.reach.mdp import (
    ReachMDPCfg,
    ActionTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
    make_default_reach_mdp_cfg,
)

__all__ = [
    "ReachTaskConfig",
    "make_reach_env",
    "make_reach_gymnasium",
    "ReachMDPCfg",
    "ActionTermCfg",
    "ObservationTermCfg",
    "RewardTermCfg",
    "TerminationTermCfg",
    "make_default_reach_mdp_cfg",
]
