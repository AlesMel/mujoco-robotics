"""URReachEnvBase — base class for the 3-D position + orientation reaching task.

This module contains the **task-level logic** shared by all control variants:
observation construction, reward computation, goal sampling, MuJoCo physics
setup, rendering, and the episode lifecycle (reset / step skeleton).

Concrete action-space variants (IK-Rel, IK-Abs, Joint-Pos) live in
dedicated sibling modules and override :meth:`_apply_action` to map
normalised ``[-1, 1]`` actions into joint-position targets.

Isaac-Lab-inspired features:

* **Last action** appended to observation for smooth control learning.
* **Observation noise** (uniform ± ``obs_noise``) on proprioceptive channels.
* **Action-rate penalty** ``-w * ||a_t - a_{t-1}||²``.
* **Curriculum** for orientation difficulty and penalty ramp-up.
* **Time-based command resampling** (Isaac Lab style).
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import gymnasium
import mujoco
import numpy as np

from mujoco_robot.robots.configs import ROBOT_CONFIGS, RobotConfig, get_robot_config
from mujoco_robot.core.ik_controller import (
    IKController,
    orientation_error_axis_angle,
    quat_error_magnitude,
    quat_multiply,
    quat_unique,
)
from mujoco_robot.core.collision import CollisionDetector
from mujoco_robot.core.xml_builder import load_robot_xml, build_reach_xml


# ---------------------------------------------------------------------------
# Dataclass for step results (non-Gymnasium usage)
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    """Container returned by :meth:`URReachEnvBase.step`."""
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict


# ---------------------------------------------------------------------------
# Base Gymnasium wrapper
# ---------------------------------------------------------------------------
class ReachGymnasiumBase(gymnasium.Env):
    """Gymnasium-compliant wrapper around a :class:`URReachEnvBase` subclass.

    Subclass this and set ``_env_cls`` to a concrete :class:`URReachEnvBase`
    subclass, or pass ``env_cls`` at construction time.  Keyword arguments
    are forwarded to the inner environment.
    """

    metadata = {"render_modes": ["rgb_array"]}

    # Subclasses override this to point to their concrete env class.
    _env_cls: type | None = None

    def __init__(
        self,
        robot: str = "ur5e",
        seed: int | None = None,
        render_mode: str | None = None,
        time_limit: int = 375,
        env_cls: type | None = None,
        **env_kwargs,
    ):
        super().__init__()
        self.render_mode = render_mode
        high_res = render_mode == "rgb_array"
        cls = env_cls or self._env_cls
        if cls is None:
            raise TypeError(
                "ReachGymnasiumBase requires either _env_cls to be set "
                "in a subclass, or env_cls passed at __init__ time."
            )
        # IsaacLab IK-Rel/Abs style default: no hold period.
        if "hold_seconds" not in env_kwargs:
            env_kwargs["hold_seconds"] = 0.0
        self.base = cls(
            robot=robot,
            render_size=(160, 120) if not high_res else (640, 480),
            time_limit=time_limit,
            seed=seed,
            **env_kwargs,
        )
        self.action_space = gymnasium.spaces.Box(
            -1.0, 1.0, shape=(self.base.action_dim,), dtype=np.float32
        )
        self.observation_space = gymnasium.spaces.Box(
            -np.inf, np.inf, shape=(self.base.observation_dim,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options=None):
        obs = self.base.reset(seed=seed)
        return obs.astype(np.float32), {}

    def step(self, action):
        res: StepResult = self.base.step(action)
        terminated = bool(res.info.get("terminated", False))
        truncated = bool(res.info.get("time_out", False))
        # Backward-compatible fallback for older info payloads.
        if not terminated and not truncated:
            time_up = (
                self.base.time_limit > 0
                and self.base.step_id >= self.base.time_limit
            )
            truncated = time_up
        return res.obs.astype(np.float32), res.reward, terminated, truncated, res.info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.base.render(mode="rgb_array")
        return None

    def close(self):
        self.base.close()


# ---------------------------------------------------------------------------
# Base core environment
# ---------------------------------------------------------------------------
class URReachEnvBase:
    """Base reaching task for a UR-style arm (task logic only).

    Supports multiple robots via the ``robot`` parameter:
        - ``"ur5e"`` — UR5e with ~0.85 m reach (default)
        - ``"ur3e"`` — UR3e with ~0.50 m reach

    Subclass and override :meth:`_apply_action` to implement a concrete
    control variant (see ``reach_env_ik_rel.py``, ``reach_env_ik_abs.py``,
    ``reach_env_joint_pos.py``).

    Observation (19 + action_dim = 25):
        ============== ===== ===================================
        Component      Dim   Description
        ============== ===== ===================================
        joint_pos        6   joint angles relative to home (noise ±0.01)
        joint_vel        6   joint velocities (noise ±0.01)
        pose_command     7   goal pose in base frame (pos_xyz, quat_wxyz)
        last_action      6   previous action (zeros at reset)
        ============== ===== ===================================
        → total = 25   (matches Isaac Lab reach env exactly)

    Reward (7 terms):
        ``-0.2 * dist``  +  ``0.16 * (1 − tanh(d/0.1))``  +
        ``-0.15 * ori_err``  +  ``0.1 * (1 − tanh(ori_err/0.2))``  +
        ``0.5 * at_goal``  +  ``-w_ar * ||Δa||²``  +  ``-w_jv * ||q̇||²``
        with curriculum on action-rate and joint-velocity penalties.
    """

    # Subclasses may refine these; defaults assume 6-DOF action.
    _action_dim: int = 6

    def __init__(
        self,
        robot: str = "ur5e",
        model_path: Optional[str] = None,
        time_limit: int = 375,
        ee_step: float = 0.06,
        ori_step: float = 0.5,
        ori_abs_max: float = np.pi,
        joint_action_scale: float = 0.1,
        reach_threshold: float = 0.05,
        ori_threshold: float = 0.35,
        terminate_on_success: bool = False,
        terminate_on_collision: bool = False,
        hold_seconds: float = 2.0,  # deprecated, kept for API compatibility
        goal_resample_time_range_s: Tuple[float, float] = (4.0, 4.0),
        goal_resample_interval: Optional[int] = None,  # deprecated
        obs_noise: float = 0.01,
        action_rate_weight: float = 0.0001,
        joint_vel_weight: float = 0.0001,
        randomize_init: bool = True,
        init_q_range: Tuple[float, float] = (0.75, 1.25),
        render_size: Tuple[int, int] = (960, 720),
        seed: Optional[int] = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)

        # --- Robot configuration ---
        cfg = get_robot_config(robot)
        self.robot = robot
        self.model_path = model_path or cfg.model_path
        self._BASE_POS = cfg.base_pos.copy()
        self._LINK_LENGTHS = cfg.link_lengths
        self._TOTAL_REACH = cfg.total_reach

        self.time_limit = time_limit
        self.ee_step = ee_step
        self.ori_step = ori_step
        self.ori_abs_max = float(ori_abs_max)
        self.joint_action_scale = joint_action_scale
        self.obs_noise = obs_noise
        self.action_rate_weight = max(action_rate_weight, 0.001)  # higher initial → less jitter
        self.joint_vel_weight = joint_vel_weight
        self.randomize_init = randomize_init
        self.init_q_range = init_q_range
        self.render_size = render_size

        # Isaac-style: command resampling based on elapsed wall-clock time.
        self.reach_threshold = reach_threshold
        self.ori_threshold = ori_threshold
        self.terminate_on_success = terminate_on_success
        self.terminate_on_collision = terminate_on_collision
        if goal_resample_interval is not None:
            # Backward compatibility for old step-count API.
            default_step_dt = 0.005 * 4
            fixed_s = float(goal_resample_interval) * default_step_dt
            self.goal_resample_time_range_s = (fixed_s, fixed_s)
        else:
            lo_s, hi_s = goal_resample_time_range_s
            lo_s = float(lo_s)
            hi_s = float(hi_s)
            if lo_s <= 0.0 or hi_s <= 0.0 or lo_s > hi_s:
                raise ValueError(
                    "goal_resample_time_range_s must be positive and ordered "
                    f"(min <= max), got {goal_resample_time_range_s}"
                )
            self.goal_resample_time_range_s = (lo_s, hi_s)
        self._goal_resample_elapsed_s = 0.0
        self._next_goal_resample_s = self.goal_resample_time_range_s[0]

        # Curriculum targets (Isaac Lab style)
        self._action_rate_target = 0.005
        self._joint_vel_target = 0.001
        self._curriculum_steps = 4500
        self._total_episodes = 0

        # Orientation curriculum
        self._ori_curriculum_start = 0.3
        self._ori_curriculum_end = np.pi
        self._ori_curriculum_steps = 6000

        # Physics / control tuning
        self.max_joint_vel = 4.0
        self.ik_damping = 0.02
        self.hold_eps = 0.05
        self.n_substeps = 4
        self.settle_steps = 300

        # Robot-specific home pose and workspace bounds
        self.init_q = cfg.init_q.copy()
        self.goal_bounds = cfg.goal_bounds.copy()
        self.ee_bounds = cfg.ee_bounds.copy()

        # ---- Build MuJoCo model ----
        robot_xml = load_robot_xml(self.model_path)
        marker_size = 0.06
        self.model_xml = build_reach_xml(robot_xml, render_size, marker_size)
        self.model = mujoco.MjModel.from_xml_string(self.model_xml)
        self.data = mujoco.MjData(self.model)

        # Solver settings
        self.model.opt.timestep = 0.005
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.iterations = max(self.model.opt.iterations, 50)
        self.model.opt.gravity[:] = [0.0, 0.0, -9.81]

        # Renderers
        self._renderer_top = mujoco.Renderer(
            self.model, height=render_size[1], width=render_size[0]
        )
        self._renderer_side = mujoco.Renderer(
            self.model, height=render_size[1], width=render_size[0]
        )
        self._renderer_ee = mujoco.Renderer(
            self.model, height=render_size[1], width=render_size[0]
        )

        # Look up IDs
        self.ee_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        self.goal_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "goal_site"
        )
        self.goal_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_cube"
        )

        self.robot_joints = [
            "shoulder_pan", "shoulder_lift", "elbow",
            "wrist1", "wrist2", "wrist3",
        ]
        self.robot_actuators = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_motor")
            for j in self.robot_joints
        ]
        self.robot_dofs = [
            self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            ]
            for j in self.robot_joints
        ]

        # Passive damping — must be high enough to prevent ringing.
        # With position-servo gain ≈ 400, critical damping requires
        # b ≈ 2·√(kp·I) ≈ 40 for typical joint inertia ≈ 1 kg·m².
        # 10.0 → damping ratio ≈ 0.25 (slightly underdamped, no visible
        # oscillation).  Friction-loss adds Coulomb stiction that kills
        # residual jitter near equilibrium.
        for dof in self.robot_dofs:
            self.model.dof_damping[dof] = max(self.model.dof_damping[dof], 10.0)
            self.model.dof_frictionloss[dof] = max(self.model.dof_frictionloss[dof], 0.5)

        # --- Self-collision detection ---
        self._collision_detector = CollisionDetector(self.model)
        self._self_collision_count = 0

        # --- IK controller ---
        self._ik = IKController(
            self.model, self.data, self.ee_site, self.robot_dofs,
            damping=self.ik_damping,
        )

        # Ensure actuator control ranges match joint limits
        for k, act in enumerate(self.robot_actuators):
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, self.robot_joints[k]
            )
            rng = self.model.jnt_range[jid]
            low, high = (-np.pi, np.pi) if rng[0] >= rng[1] else rng
            self.model.actuator_ctrlrange[act] = [low, high]
            if self.model.actuator_gainprm[act, 0] <= 0.0:
                self.model.actuator_gainprm[act, 0] = 400.0

        # State
        self._last_targets = self.init_q.copy()
        self.goal_pos = np.zeros(3)
        self.goal_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._home_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.step_id = 0
        self._init_dist = 1.0
        self._init_ori_err = np.pi
        self._goals_reached = 0
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)

    # -------------------------------------------------------------- Properties
    @property
    def action_dim(self) -> int:
        """Action dimensionality (overridable by subclasses)."""
        return self._action_dim

    @property
    def observation_dim(self) -> int:
        """Observation dimensionality: 19 base + action_dim (last_action).

        Matches Isaac Lab reach env: joint_pos(6) + joint_vel(6) +
        pose_command(7) + last_action(action_dim).
        """
        return 19 + self.action_dim

    # -------------------------------------------------------------- Reset
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment and return the initial observation."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.step_id = 0
        self._goals_reached = 0
        self._self_collision_count = 0
        self._goal_resample_elapsed_s = 0.0
        self._next_goal_resample_s = self._sample_goal_resample_time_s()
        self._last_targets = self.init_q.copy()
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)

        # Set robot to home pose (with optional Isaac Lab-style randomization)
        self._total_episodes += 1
        for qi, j in enumerate(self.robot_joints):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            q_home = self.init_q[qi]
            if self.randomize_init:
                scale = float(self._rng.uniform(*self.init_q_range))
                q_home = q_home * scale
                lo, hi = self.model.jnt_range[jid]
                if lo < hi:
                    q_home = float(np.clip(q_home, lo, hi))
            self.data.qpos[self.model.jnt_qposadr[jid]] = q_home
            self.data.qvel[self.model.jnt_dofadr[jid]] = 0.0
            act_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_motor"
            )
            if act_id != -1:
                self.data.ctrl[act_id] = q_home
        # Update last targets to match randomized start
        for qi, j in enumerate(self.robot_joints):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            self._last_targets[qi] = self.data.qpos[self.model.jnt_qposadr[jid]]

        mujoco.mj_forward(self.model, self.data)

        # Store the home orientation for orientation curriculum
        self._home_quat = self._ee_quat()

        # Sample goal
        self.goal_pos = self._sample_goal()
        self.goal_quat = self._sample_goal_quat()
        self._place_goal_marker(self.goal_pos, self.goal_quat)

        for _ in range(self.settle_steps):
            mujoco.mj_step(self.model, self.data)

        self._init_dist = self._ee_goal_dist()
        self._init_ori_err = self._orientation_error_magnitude()
        return self._observe()

    # -------------------------------------------------------------- Goal sampling
    def _sample_goal(self) -> np.ndarray:
        """Sample a random reachable goal position."""
        cfg = get_robot_config(self.robot)
        ee_pos = self.data.site_xpos[self.ee_site].copy()
        min_base, max_base = cfg.goal_distance
        min_height = cfg.goal_min_height
        min_ee = cfg.goal_min_ee_dist

        for _ in range(500):
            goal = np.array([
                self._rng.uniform(self.goal_bounds[0, 0], self.goal_bounds[0, 1]),
                self._rng.uniform(self.goal_bounds[1, 0], self.goal_bounds[1, 1]),
                self._rng.uniform(self.goal_bounds[2, 0], self.goal_bounds[2, 1]),
            ])
            if goal[2] < min_height:
                continue
            dist_from_base = np.linalg.norm(goal - self._BASE_POS)
            if dist_from_base < min_base or dist_from_base > max_base:
                continue
            if np.linalg.norm(goal - ee_pos) < min_ee:
                continue
            return goal
        return self._BASE_POS + np.array([0.25, 0.0, 0.30])

    def _sample_goal_quat(self) -> np.ndarray:
        """Sample a random goal orientation with curriculum."""
        progress = min(1.0, self._total_episodes / max(1, self._ori_curriculum_steps))
        max_angle = (
            self._ori_curriculum_start
            + (self._ori_curriculum_end - self._ori_curriculum_start) * progress
        )

        axis = self._rng.standard_normal(3)
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = axis / norm

        angle = self._rng.uniform(0, max_angle)

        half = angle / 2.0
        dq = np.array([
            np.cos(half),
            axis[0] * np.sin(half),
            axis[1] * np.sin(half),
            axis[2] * np.sin(half),
        ])

        home_quat = self._home_quat
        goal_q = quat_multiply(dq, home_quat)
        return quat_unique(goal_q)

    def _place_goal_marker(self, pos: np.ndarray, quat: np.ndarray) -> None:
        """Move and rotate the goal marker to the desired pose."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_body")
        self.model.body_pos[body_id] = pos
        self.model.body_quat[body_id] = quat
        mujoco.mj_forward(self.model, self.data)

    # -------------------------------------------------------------- IK + control helpers
    def _ee_quat(self) -> np.ndarray:
        """Current EE orientation as unit quaternion (w,x,y,z)."""
        return self._ik.ee_quat()

    def _desired_ee_relative(
        self,
        delta_xyz: np.ndarray,
        delta_ori: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute desired EE pose from **relative** Cartesian action."""
        pos = self.data.site_xpos[self.ee_site].copy() + delta_xyz
        for i in range(3):
            pos[i] = float(np.clip(pos[i], self.ee_bounds[i, 0], self.ee_bounds[i, 1]))

        current_quat = self._ee_quat()
        angle = np.linalg.norm(delta_ori)
        if angle > 1e-8:
            axis = delta_ori / angle
            half = angle / 2.0
            dq = np.array([
                np.cos(half),
                axis[0] * np.sin(half),
                axis[1] * np.sin(half),
                axis[2] * np.sin(half),
            ])
            target_quat = quat_unique(quat_multiply(dq, current_quat))
        else:
            target_quat = current_quat

        return pos, target_quat.copy()

    def _desired_ee_absolute(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute desired EE pose from **absolute** Cartesian action.

        Action layout (all in [-1, 1]):
            - ``action[:3]`` maps linearly to workspace XYZ bounds.
            - ``action[3:6]`` is an absolute axis-angle vector scaled by
              ``ori_abs_max`` and applied about ``_home_quat``.
        """
        lo = self.ee_bounds[:, 0]
        hi = self.ee_bounds[:, 1]
        pos = lo + 0.5 * (action[:3] + 1.0) * (hi - lo)

        aa = action[3:6] * self.ori_abs_max
        angle = np.linalg.norm(aa)
        if angle > self.ori_abs_max and angle > 1e-8:
            aa = aa * (self.ori_abs_max / angle)
            angle = self.ori_abs_max

        if angle > 1e-8:
            axis = aa / angle
            half = angle / 2.0
            dq = np.array([
                np.cos(half),
                axis[0] * np.sin(half),
                axis[1] * np.sin(half),
                axis[2] * np.sin(half),
            ])
            target_quat = quat_unique(quat_multiply(dq, self._home_quat))
        else:
            target_quat = self._home_quat.copy()

        return pos.astype(float), target_quat

    def _ik_cartesian(self, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        return self._ik.solve(target_pos, target_quat)

    def _clamp_to_limits(self, targets: np.ndarray) -> np.ndarray:
        out = targets.copy()
        for k, j in enumerate(self.robot_joints):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            lo, hi = self.model.jnt_range[jid]
            if lo < hi:
                out[k] = float(np.clip(out[k], lo, hi))
        return out

    # -------------------------------------------------------------- Observation
    def _ee_goal_dist(self) -> float:
        return float(np.linalg.norm(
            self.data.site_xpos[self.ee_site] - self.goal_pos
        ))

    def _orientation_error(self) -> np.ndarray:
        """Axis-angle orientation error vector (3-D), from EE toward goal."""
        return orientation_error_axis_angle(self._ee_quat(), self.goal_quat)

    def _orientation_error_magnitude(self) -> float:
        """Scalar orientation error in radians ∈ [0, π]."""
        return quat_error_magnitude(self._ee_quat(), self.goal_quat)

    def _observe(self) -> np.ndarray:
        """Build Isaac-Lab-compatible observation vector.

        Layout (25-D for 6-DOF action):
            joint_pos_rel (6) + joint_vel_rel (6) + pose_command (7) + last_action (6)

        ``pose_command`` is the goal pose expressed in the robot **base frame**
        as ``[x, y, z, qw, qx, qy, qz]``, matching Isaac Lab's
        ``generated_commands("ee_pose")`` observation term.
        """
        parts: List[np.ndarray] = []

        # 1. Joint positions — relative to home pose  (6,)
        for qi, j in enumerate(self.robot_joints):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            qpos = self.data.qpos[self.model.jnt_qposadr[jid]]
            parts.append(np.array([qpos - self.init_q[qi]]))

        # 2. Joint velocities  (6,)
        for j in self.robot_joints:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            parts.append(np.array([self.data.qvel[self.model.jnt_dofadr[jid]]]))

        # 3. Pose command in robot base frame  (7,)
        #    Isaac Lab stores this as (x, y, z, qw, qx, qy, qz) relative to
        #    the robot root.  For our fixed-base arm the base frame origin is
        #    self._BASE_POS with identity orientation.
        goal_pos_base = self.goal_pos - self._BASE_POS              # (3,)
        parts.append(goal_pos_base)
        parts.append(self.goal_quat.copy())                          # (4,) wxyz

        # 4. Last action  (action_dim,)
        parts.append(self._last_action.copy())

        obs = np.concatenate(parts).astype(np.float32)

        # Observation noise on proprioceptive channels (first 12: joint_pos + joint_vel)
        # Matches Isaac Lab: Unoise(n_min=-0.01, n_max=0.01) on both terms.
        if self.obs_noise > 0.0:
            noise = self._rng.uniform(
                -self.obs_noise, self.obs_noise, size=12
            ).astype(np.float32)
            obs[:12] += noise

        return obs

    # -------------------------------------------------------------- Reward
    def _curriculum_weight(self, base: float, target: float) -> float:
        """Linearly ramp a penalty weight from *base* to *target*."""
        progress = min(1.0, self._total_episodes / max(1, self._curriculum_steps))
        return base + (target - base) * progress

    def _resample_goal(self) -> None:
        """Sample a fresh goal and update the marker (mid-episode)."""
        self.goal_pos = self._sample_goal()
        self.goal_quat = self._sample_goal_quat()
        self._place_goal_marker(self.goal_pos, self.goal_quat)

    def _sample_goal_resample_time_s(self) -> float:
        """Draw next command duration from ``goal_resample_time_range_s``."""
        lo_s, hi_s = self.goal_resample_time_range_s
        if hi_s <= lo_s:
            return lo_s
        return float(self._rng.uniform(lo_s, hi_s))

    def _maybe_resample_goal(self) -> bool:
        """Advance command timer and resample when the sampled duration elapses."""
        step_dt = self.model.opt.timestep * self.n_substeps
        self._goal_resample_elapsed_s += step_dt
        if self._goal_resample_elapsed_s < self._next_goal_resample_s:
            return False

        self._goal_resample_elapsed_s = 0.0
        self._next_goal_resample_s = self._sample_goal_resample_time_s()
        self._goals_reached += 1
        self._resample_goal()
        return True

    def _compute_done_flags(self, dist: float, ori_err_mag: float) -> tuple[bool, bool, bool, bool, bool]:
        """Compute Manager-style done flags: success/failure + timeout."""
        success = dist < self.reach_threshold and ori_err_mag < self.ori_threshold
        failure = self._self_collision_count > 0
        terminated = (
            (self.terminate_on_success and success)
            or (self.terminate_on_collision and failure)
        )
        time_up = self.time_limit > 0 and (self.step_id + 1) >= self.time_limit
        done = terminated or time_up
        return success, failure, terminated, time_up, done

    def _compute_reward(self) -> Tuple[float, bool, Dict]:
        """Dense reward (7 terms), with Isaac-style time-based goal resampling."""
        dist = self._ee_goal_dist()
        ori_err_mag = self._orientation_error_magnitude()

        goal_resampled = self._maybe_resample_goal()

        # --- Reward terms ---
        pos_reward_coarse = -0.2 * dist
        pos_reward_fine = 0.16 * (1.0 - math.tanh(dist / 0.1))
        ori_reward = -0.15 * ori_err_mag
        ori_reward_fine = 0.1 * (1.0 - math.tanh(ori_err_mag / 0.2))
        at_goal_bonus = 0.5 if (dist < self.reach_threshold and ori_err_mag < self.ori_threshold) else 0.0

        reward = (pos_reward_coarse + pos_reward_fine
                  + ori_reward + ori_reward_fine + at_goal_bonus)

        cur_ar_w = self._curriculum_weight(
            self.action_rate_weight, self._action_rate_target
        )
        action_delta = self._last_action - self._prev_action
        reward -= cur_ar_w * float(np.dot(action_delta, action_delta))

        cur_jv_w = self._curriculum_weight(
            self.joint_vel_weight, self._joint_vel_target
        )
        jvel_sq = 0.0
        for dof in self.robot_dofs:
            jvel_sq += self.data.qvel[dof] ** 2
        reward -= cur_jv_w * jvel_sq

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
            "goal_resampled": goal_resampled,
            "goals_reached": self._goals_reached,
            "self_collisions": self._self_collision_count,
            "ee_pos": self.data.site_xpos[self.ee_site].copy(),
            "ee_quat": self._ee_quat(),
            "goal_pos": self.goal_pos.copy(),
            "goal_quat": self.goal_quat.copy(),
        }
        return float(reward), done, info

    # -------------------------------------------------------------- Action hook
    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        """Convert normalised [-1, 1] action to joint-position targets.

        Subclasses **must** override this method.

        Parameters
        ----------
        action : (action_dim,) array
            Clipped action in [-1, 1].

        Returns
        -------
        qpos_targets : (n_joints,) array
            Desired joint positions to command to actuators.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _apply_action(). "
            "Use a concrete variant: ReachIKRelEnv, ReachIKAbsEnv, "
            "or ReachJointPosEnv."
        )

    # -------------------------------------------------------------- Step
    def step(self, action: Iterable[float]) -> StepResult:
        """Execute one environment step.

        Parameters
        ----------
        action : array-like, shape (action_dim,)
            Action in [-1, 1].  Interpretation depends on the variant.

        Returns
        -------
        StepResult
            Named result with ``obs``, ``reward``, ``done``, ``info``.
        """
        act = np.asarray(action, dtype=float).flatten()
        if act.shape[0] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {act.shape[0]}")
        act = np.clip(act, -1.0, 1.0)

        self._prev_action = self._last_action.copy()
        self._last_action = act.astype(np.float32).copy()

        prev_targets = self._last_targets.copy()

        # Delegate to variant-specific action mapping
        qpos_targets = self._apply_action(act)
        qpos_targets = self._clamp_to_limits(qpos_targets)
        self._last_targets = qpos_targets.copy()

        for k, act_id in enumerate(self.robot_actuators):
            self.data.ctrl[act_id] = qpos_targets[k]

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        # Detect self-collision
        self._self_collision_count = self._collision_detector.count(self.data)

        # Revert on collision
        if self._self_collision_count > 0:
            self._last_targets = prev_targets
            for k, act_id in enumerate(self.robot_actuators):
                self.data.ctrl[act_id] = prev_targets[k]
            for dof in self.robot_dofs:
                self.data.qvel[dof] = 0.0
            mujoco.mj_forward(self.model, self.data)
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.model, self.data)

        reward, done, info = self._compute_reward()
        self.step_id += 1
        return StepResult(self._observe(), reward, done, info)

    # -------------------------------------------------------------- Render
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the scene."""
        if mode == "rgb_array":
            self._renderer_top.update_scene(self.data, camera="top")
            frame_top = self._renderer_top.render()
            self._renderer_side.update_scene(self.data, camera="side")
            frame_side = self._renderer_side.render()
            self._renderer_ee.update_scene(self.data, camera="ee_cam")
            frame_ee = self._renderer_ee.render()
            return np.concatenate([frame_top, frame_side, frame_ee], axis=1)
        return None

    def close(self) -> None:
        """Release rendering resources."""
        for r in (self._renderer_top, self._renderer_side, self._renderer_ee):
            if r is not None:
                if hasattr(r, "close"):
                    r.close()
                elif hasattr(r, "free"):
                    r.free()
        self._renderer_top = None
        self._renderer_side = None
        self._renderer_ee = None

    def sample_action(self) -> np.ndarray:
        """Sample a random action."""
        return self._rng.uniform(-1, 1, size=self.action_dim).astype(np.float32)
