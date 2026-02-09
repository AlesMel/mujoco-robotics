"""ReachJointPosEnv — Joint-position control variant for the reach task.

Action (6-D): ``[dq1, dq2, dq3, dq4, dq5, dq6]`` in ``[-1, 1]``.
Each component is scaled by ``joint_action_scale`` and added to the
**current actual joint positions** (not the last commanded target).
This matches Isaac Lab's ``RelativeJointPositionAction``:

    target = current_joint_pos + action × scale

Usage::

    from mujoco_robot.envs.reach import ReachJointPosEnv, ReachJointPosGymnasium

    env = ReachJointPosEnv(robot="ur3e")
    obs = env.reset()
    result = env.step(env.sample_action())

    gym_env = ReachJointPosGymnasium(robot="ur3e")
"""
from __future__ import annotations

import mujoco
import numpy as np

from mujoco_robot.envs.reach.reach_env_base import (
    ReachGymnasiumBase,
    URReachEnvBase,
)


class ReachJointPosEnv(URReachEnvBase):
    """Reach task with **relative joint-position** actions (Isaac Lab style).

    Action layout (all in [-1, 1]):
        ========  ============================================
        Index     Description
        ========  ============================================
        0–5       Joint-position offsets (×joint_action_scale)
        ========  ============================================

    The offset is applied relative to the **current** measured joint
    positions, *not* the previously commanded targets.  This matches
    Isaac Lab's ``RelativeJointPositionAction`` and prevents target
    drift when the actuator cannot perfectly track the command.
    """

    _action_dim = 6

    def _current_joint_pos(self) -> np.ndarray:
        """Read the actual joint positions from the MuJoCo state."""
        q = np.empty(len(self.robot_joints))
        for k, j in enumerate(self.robot_joints):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            q[k] = self.data.qpos[self.model.jnt_qposadr[jid]]
        return q

    # EMA blending factor for target smoothing.  Lower values produce
    # smoother motion but slower response.  0.3 gives ~95 % convergence
    # in ≈ 9 steps (≈ 0.18 s at 50 Hz control rate).
    _ema_alpha: float = 0.3

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        # Near-zero actions → hold in place (matches IK-rel hold_eps)
        if np.linalg.norm(action) < self.hold_eps:
            return self._last_targets.copy()
        # Isaac Lab: target = current_joint_pos + action * scale
        raw_target = self._current_joint_pos() + action * self.joint_action_scale
        # EMA smoothing prevents oscillation near goal (analogous to
        # Isaac Lab's EMAJointPositionToLimitsAction).
        return self._ema_alpha * raw_target + (1.0 - self._ema_alpha) * self._last_targets


class ReachJointPosGymnasium(ReachGymnasiumBase):
    """Gymnasium wrapper for :class:`ReachJointPosEnv`."""

    _env_cls = ReachJointPosEnv
