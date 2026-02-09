"""Joint-position reach variant with Isaac Lab reward terms.

This environment keeps the same action mapping as :class:`ReachJointPosEnv`
and only changes reward computation to match Isaac Lab's Franka joint-position
reach task reward terms exactly:

1. ``position_command_error_tanh`` with weight ``+0.1`` and std ``0.1``
2. ``orientation_command_error`` with weight ``-0.1``
3. ``action_rate_l2`` with weight ``-1e-4``
4. ``joint_vel_l2`` with weight ``-1e-4``

As in Isaac Lab's reward manager, the weighted sum is scaled by the control
step duration ``step_dt``.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from mujoco_robot.envs.reach.reach_env_base import ReachGymnasiumBase
from mujoco_robot.envs.reach.reach_env_joint_pos import ReachJointPosEnv


ISAAC_CONTROL_DT_S = 0.005 * 4
ISAAC_EPISODE_SECONDS = 3.0
ISAAC_EPISODE_STEPS = int(round(ISAAC_EPISODE_SECONDS / ISAAC_CONTROL_DT_S))
ISAAC_GOAL_RESAMPLE_TIME_RANGE_S = (4.0, 4.0)
ISAAC_JOINT_ACTION_SCALE = 0.25
ISAAC_ACTION_DEADZONE = 0.0
ISAAC_EMA_ALPHA = 1.0


class ReachJointPosIsaacRewardEnv(ReachJointPosEnv):
    """Reach task with joint-position control and Isaac Lab reward.

    Defaults are configured to mirror common Isaac Lab reach settings:
    - episode length ~= 3.0 s
    - command resampling window = [4.0, 4.0] s
    - direct relative joint-position targets (no EMA smoothing)
    - no action deadzone
    - no task-based termination by default (timeout only)
    """

    DEFAULT_TIME_LIMIT = ISAAC_EPISODE_STEPS
    _ema_alpha = ISAAC_EMA_ALPHA

    def __init__(self, robot: str = "ur5e", **kwargs) -> None:
        kwargs.setdefault("time_limit", self.DEFAULT_TIME_LIMIT)
        kwargs.setdefault("goal_resample_time_range_s", ISAAC_GOAL_RESAMPLE_TIME_RANGE_S)
        kwargs.setdefault("joint_action_scale", ISAAC_JOINT_ACTION_SCALE)
        kwargs.setdefault("terminate_on_success", False)
        kwargs.setdefault("terminate_on_collision", False)
        kwargs.setdefault("hold_seconds", 0.0)
        super().__init__(robot=robot, **kwargs)
        # Isaac action term has no deadzone around zero.
        self.hold_eps = ISAAC_ACTION_DEADZONE

    def _compute_reward(self) -> Tuple[float, bool, Dict]:
        dist = self._ee_goal_dist()
        ori_err_mag = self._orientation_error_magnitude()

        goal_resampled = self._maybe_resample_goal()

        # Isaac Lab reward terms (joint_pos_env_cfg.RewardsCfg).
        pos_l2 = float(dist)                 # position_command_error (L2 distance)
        pos_tanh = 1.0 - math.tanh(dist / 0.1)
        action_delta = self._last_action - self._prev_action
        action_rate_l2 = float(np.dot(action_delta, action_delta))
        joint_vel = self.data.qvel[self.robot_dofs]
        joint_vel_l2 = float(np.dot(joint_vel, joint_vel))

        reward = (
            - 0.2 * pos_l2
            + 0.1 * pos_tanh
            - 0.1 * ori_err_mag
            - 1.0e-4 * action_rate_l2
            - 1.0e-4 * joint_vel_l2
        )

        # Isaac Lab RewardManager integrates rewards over environment step dt.
        step_dt = self.model.opt.timestep * self.n_substeps
        reward *= step_dt

        success, failure, terminated, time_up, done = self._compute_done_flags(dist, ori_err_mag)

        info = {
            "dist": dist,
            "ori_err": ori_err_mag,
            "success": success,
            "failure": failure,
            "terminated": terminated,
            "time_out": time_up,
            "goal_resample_elapsed_s": self._goal_resample_elapsed_s,
            "goal_resample_target_s": self._next_goal_resample_s,
            "reward_terms": {
                "position_command_error_tanh": pos_tanh,
                "orientation_command_error": ori_err_mag,
                "action_rate_l2": action_rate_l2,
                "joint_vel_l2": joint_vel_l2,
                "step_dt": step_dt,
            },
            "goal_resampled": goal_resampled,
            "goals_reached": self._goals_reached,
            "self_collisions": self._self_collision_count,
            "ee_pos": self.data.site_xpos[self.ee_site].copy(),
            "ee_quat": self._ee_quat(),
            "goal_pos": self.goal_pos.copy(),
            "goal_quat": self.goal_quat.copy(),
        }
        return float(reward), done, info


class ReachJointPosIsaacRewardGymnasium(ReachGymnasiumBase):
    """Gymnasium wrapper for :class:`ReachJointPosIsaacRewardEnv`."""

    _env_cls = ReachJointPosIsaacRewardEnv

    def __init__(
        self,
        robot: str = "ur5e",
        seed: int | None = None,
        render_mode: str | None = None,
        time_limit: int = ISAAC_EPISODE_STEPS,
        **env_kwargs,
    ):
        super().__init__(
            robot=robot,
            seed=seed,
            render_mode=render_mode,
            time_limit=time_limit,
            **env_kwargs,
        )
