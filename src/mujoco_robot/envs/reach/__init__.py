"""Reach task — modular action-space variants (Isaac Lab style).

Each control variant lives in its own module:

* :mod:`.reach_env_joint_pos` — Joint-Position (default)
* :mod:`.reach_env_ik_rel` — IK-Relative
* :mod:`.reach_env_ik_abs` — IK-Absolute
* :mod:`.reach_env_joint_pos_isaac_reward` — Joint-Position + Isaac reward

All variants share the same base task logic from
:mod:`.reach_env_base`.

Quick reference::

    from mujoco_robot.envs.reach import (
        # Base (for custom subclasses)
        URReachEnvBase,
        ReachGymnasiumBase,
        StepResult,
        # IK-Relative
        ReachIKRelEnv,
        ReachIKRelGymnasium,
        # IK-Absolute
        ReachIKAbsEnv,
        ReachIKAbsGymnasium,
        # Joint-Position
        ReachJointPosEnv,
        ReachJointPosGymnasium,
        ReachJointPosIsaacRewardEnv,
        ReachJointPosIsaacRewardGymnasium,
    )
"""

from mujoco_robot.envs.reach.reach_env_base import (
    URReachEnvBase,
    ReachGymnasiumBase,
    StepResult,
)
from mujoco_robot.envs.reach.reach_env_ik_rel import (
    ReachIKRelEnv,
    ReachIKRelGymnasium,
)
from mujoco_robot.envs.reach.reach_env_ik_abs import (
    ReachIKAbsEnv,
    ReachIKAbsGymnasium,
)
from mujoco_robot.envs.reach.reach_env_joint_pos import (
    ReachJointPosEnv,
    ReachJointPosGymnasium,
)
from mujoco_robot.envs.reach.reach_env_joint_pos_isaac_reward import (
    ReachJointPosIsaacRewardEnv,
    ReachJointPosIsaacRewardGymnasium,
)
from mujoco_robot.tasks.manager_based.manipulation.reach.mdp import (
    ActionTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
    ReachMDPCfg,
    make_default_reach_mdp_cfg,
)

# Variant registry — maps string keys to (env_cls, gymnasium_cls) pairs
REACH_VARIANTS = {
    "ik_rel": (ReachIKRelEnv, ReachIKRelGymnasium),
    "ik_abs": (ReachIKAbsEnv, ReachIKAbsGymnasium),
    "joint_pos": (ReachJointPosEnv, ReachJointPosGymnasium),
    "joint_pos_isaac_reward": (
        ReachJointPosIsaacRewardEnv,
        ReachJointPosIsaacRewardGymnasium,
    ),
}

__all__ = [
    "URReachEnvBase",
    "ReachGymnasiumBase",
    "StepResult",
    "ReachIKRelEnv",
    "ReachIKRelGymnasium",
    "ReachIKAbsEnv",
    "ReachIKAbsGymnasium",
    "ReachJointPosEnv",
    "ReachJointPosGymnasium",
    "ReachJointPosIsaacRewardEnv",
    "ReachJointPosIsaacRewardGymnasium",
    "ActionTermCfg",
    "ObservationTermCfg",
    "RewardTermCfg",
    "TerminationTermCfg",
    "ReachMDPCfg",
    "make_default_reach_mdp_cfg",
    "REACH_VARIANTS",
]
