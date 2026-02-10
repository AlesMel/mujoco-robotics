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

import numpy as np

from mujoco_robot.envs.reach.reach_env_base import (
    ReachGymnasiumBase,
    URReachEnvBase,
)
from mujoco_robot.envs.reach.mdp import actions


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
        q = np.empty(len(self._robot_qpos_ids))
        for i, qpos_adr in enumerate(self._robot_qpos_ids):
            q[i] = self.data.qpos[qpos_adr]
        return q

    # IsaacLab JointPositionAction behavior is direct target updates.
    _ema_alpha: float = 1.0

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        return actions.joint_relative_targets(self, action)


class ReachJointPosGymnasium(ReachGymnasiumBase):
    """Gymnasium wrapper for :class:`ReachJointPosEnv`."""

    _env_cls = ReachJointPosEnv
