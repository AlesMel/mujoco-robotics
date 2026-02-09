"""ReachIKAbsEnv — IK-Absolute control variant for the reach task.

Action (6-D): ``[x, y, z, ax, ay, az]`` in ``[-1, 1]``.
XYZ components map linearly to the workspace bounds (absolute target
position).  Rotation components are an absolute axis-angle vector
scaled by ``ori_abs_max``, applied about the home orientation.

Usage::

    from mujoco_robot.envs.reach import ReachIKAbsEnv, ReachIKAbsGymnasium

    env = ReachIKAbsEnv(robot="ur3e")
    obs = env.reset()
    result = env.step(env.sample_action())

    gym_env = ReachIKAbsGymnasium(robot="ur3e")
"""
from __future__ import annotations

import numpy as np

from mujoco_robot.envs.reach.reach_env_base import (
    ReachGymnasiumBase,
    URReachEnvBase,
)
from mujoco_robot.envs.reach.mdp import actions


class ReachIKAbsEnv(URReachEnvBase):
    """Reach task with **absolute Cartesian IK** actions (Isaac Lab IK-Abs).

    Action layout (all in [-1, 1]):
        ========  ==============================================
        Index     Description
        ========  ==============================================
        0–2       Absolute XYZ mapped to workspace bounds
        3–5       Absolute axis-angle (×ori_abs_max) about home
        ========  ==============================================
    """

    _action_dim = 6

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        return actions.ik_absolute_joint_targets(self, action)


class ReachIKAbsGymnasium(ReachGymnasiumBase):
    """Gymnasium wrapper for :class:`ReachIKAbsEnv`."""

    _env_cls = ReachIKAbsEnv
