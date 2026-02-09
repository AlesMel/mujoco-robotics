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
        target_pos, target_quat = self._desired_ee_absolute(action)

        qvel_cmd = self._ik_cartesian(target_pos, target_quat)
        qvel_cmd = np.clip(qvel_cmd, -self.max_joint_vel, self.max_joint_vel)

        ik_gain = 0.35
        return self._last_targets + qvel_cmd * ik_gain


class ReachIKAbsGymnasium(ReachGymnasiumBase):
    """Gymnasium wrapper for :class:`ReachIKAbsEnv`."""

    _env_cls = ReachIKAbsEnv
