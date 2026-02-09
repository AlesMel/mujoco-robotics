"""Manager-based MDP API for reach tasks."""

from mujoco_robot.envs.reach.mdp.terms import (
    ActionTermCfg,
    ObservationTermCfg,
    ReachMDPCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mujoco_robot.envs.reach.mdp.managers import (
    ActionManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
)
from mujoco_robot.envs.reach.mdp.config import (
    default_observation_terms,
    default_reward_terms,
    default_success_term,
    default_failure_term,
    default_timeout_term,
    make_default_reach_mdp_cfg,
)
from . import actions, observations, rewards, terminations

__all__ = [
    "ActionTermCfg",
    "ObservationTermCfg",
    "RewardTermCfg",
    "TerminationTermCfg",
    "ReachMDPCfg",
    "ActionManager",
    "ObservationManager",
    "RewardManager",
    "TerminationManager",
    "default_observation_terms",
    "default_reward_terms",
    "default_success_term",
    "default_failure_term",
    "default_timeout_term",
    "make_default_reach_mdp_cfg",
    "actions",
    "observations",
    "rewards",
    "terminations",
]
