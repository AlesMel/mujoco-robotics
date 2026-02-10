"""Gymnasium-ready environments for UR robot tasks.

Reach variants (Isaac Lab style — each action space is a separate class):

* :class:`ReachIKRelEnv` / :class:`ReachIKRelGymnasium` — IK-Relative
* :class:`ReachIKAbsEnv` / :class:`ReachIKAbsGymnasium` — IK-Absolute
* :class:`ReachJointPosEnv` / :class:`ReachJointPosGymnasium` — Joint-Position
* :class:`ReachJointPosIsaacRewardEnv` / :class:`ReachJointPosIsaacRewardGymnasium`
  — Joint-Position with Isaac reward terms
"""

# --- Modular reach variants (preferred) ---
from mujoco_robot.envs.reach import (
    URReachEnvBase,
    ReachGymnasiumBase,
    StepResult,
    ReachIKRelEnv,
    ReachIKRelGymnasium,
    ReachIKAbsEnv,
    ReachIKAbsGymnasium,
    ReachJointPosEnv,
    ReachJointPosGymnasium,
    ReachJointPosIsaacRewardEnv,
    ReachJointPosIsaacRewardGymnasium,
    REACH_VARIANTS,
)

# --- Slot sorter ---
from mujoco_robot.envs.slot_sorter import URSlotSorterEnv, SlotSorterGymnasium

__all__ = [
    # Base
    "URReachEnvBase",
    "ReachGymnasiumBase",
    "StepResult",
    # Modular variants
    "ReachIKRelEnv",
    "ReachIKRelGymnasium",
    "ReachIKAbsEnv",
    "ReachIKAbsGymnasium",
    "ReachJointPosEnv",
    "ReachJointPosGymnasium",
    "ReachJointPosIsaacRewardEnv",
    "ReachJointPosIsaacRewardGymnasium",
    "REACH_VARIANTS",
    # Slot sorter
    "URSlotSorterEnv",
    "SlotSorterGymnasium",
]
