"""URReachEnv — 3-D position + yaw reaching task for RL.

The task: move the end-effector of a 6-DOF UR arm to a random
goal position **and** match a target yaw orientation.  A red cube
with RGB coordinate axes shows the target pose; a matching set of
axes on the EE shows the current orientation.

Two **action modes** are supported (selected via ``action_mode``):

* ``"cartesian"`` (default) — 4-D ``[dx, dy, dz, dyaw]`` Cartesian
  velocity commands processed through a damped-least-squares IK solver.
* ``"joint"`` — 6-D joint-position offsets (like Isaac Lab).  Each
  action is scaled by ``joint_action_scale`` and added to the current
  joint-position targets.

Isaac-Lab-inspired features:

* **Last action** is appended to the observation vector so the policy
  can learn smooth behaviours.
* **Observation noise** (uniform ± ``obs_noise``) is added to
  proprioceptive channels for sim-to-real robustness.
* **Action-rate penalty** ``-action_rate_weight * ||a_t - a_{t-1}||²``
  discourages jerky motions.

RL usage (Gymnasium)::

    from mujoco_robot.envs import ReachGymnasium
    env = ReachGymnasium()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    # Joint-space actions:
    env = ReachGymnasium(action_mode="joint")

Teleop::

    from mujoco_robot.envs import URReachEnv
    from mujoco_robot.teleop import ReachTeleop
    env = URReachEnv(robot="ur5e", time_limit=0)
    ReachTeleop(env).run()
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
from mujoco_robot.core.ik_controller import IKController
from mujoco_robot.core.collision import CollisionDetector
from mujoco_robot.core.xml_builder import load_robot_xml, build_reach_xml


# ---------------------------------------------------------------------------
# Dataclass for step results (non-Gymnasium usage)
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    """Container returned by :meth:`URReachEnv.step`."""
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict


# ---------------------------------------------------------------------------
# Gymnasium wrapper (for SB3 / VecEnv)
# ---------------------------------------------------------------------------
class ReachGymnasium(gymnasium.Env):
    """Gymnasium-compliant wrapper around :class:`URReachEnv`.

    Parameters
    ----------
    robot : str
        Robot model name (``"ur5e"`` or ``"ur3e"``).
    seed : int | None
        Random seed for reproducibility.
    render : bool
        If ``True``, use high-resolution rendering (for video capture).
    time_limit : int
        Maximum steps per episode (0 = unlimited).
    action_mode : str
        ``"cartesian"`` (4-D IK) or ``"joint"`` (6-D joint offsets).
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        robot: str = "ur5e",
        seed: int | None = None,
        render: bool = False,
        time_limit: int = 1000,
        action_mode: str = "cartesian",
    ):
        super().__init__()
        self.base = URReachEnv(
            robot=robot,
            render_size=(160, 120) if not render else (640, 480),
            time_limit=time_limit,
            seed=seed,
            action_mode=action_mode,
        )
        self.action_space = gymnasium.spaces.Box(
            -1.0, 1.0, shape=(self.base.action_dim,), dtype=np.float32
        )
        self.observation_space = gymnasium.spaces.Box(
            -np.inf, np.inf, shape=(self.base.observation_dim,), dtype=np.float32
        )
        self.render_mode = "rgb_array" if render else None

    def reset(self, *, seed: int | None = None, options=None):
        obs = self.base.reset(seed=seed)
        return obs.astype(np.float32), {}

    def step(self, action):
        res: StepResult = self.base.step(action)
        reached = res.info.get("reached", False)
        time_up = (
            self.base.time_limit > 0
            and self.base.step_id >= self.base.time_limit
        )
        terminated = reached
        truncated = time_up and not reached
        return res.obs.astype(np.float32), res.reward, terminated, truncated, res.info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.base.render(mode="rgb_array")
        return None

    def close(self):
        self.base.close()


# ---------------------------------------------------------------------------
# Core environment
# ---------------------------------------------------------------------------
class URReachEnv:
    """Simple position-reaching task for a UR-style arm.

    Supports multiple robots via the ``robot`` parameter:
        - ``"ur5e"`` — UR5e with ~0.85 m reach (default)
        - ``"ur3e"`` — UR3e with ~0.50 m reach (smaller workspace)

    Action modes:
        - ``"cartesian"`` (4-D): ``[dx, dy, dz, dyaw]`` via IK.
        - ``"joint"`` (6-D): joint-position offsets × ``joint_action_scale``.

    Observation (25 + action_dim = 29 or 31):
        ============== ===== ===================================
        Component      Dim   Description
        ============== ===== ===================================
        joint_pos        6   robot joint angles (relative to home)
        joint_vel        6   robot joint velocities
        ee_pos           3   end-effector world position
        ee_yaw           1   end-effector yaw angle
        goal_pos         3   target world position
        goal_yaw         1   target yaw angle
        goal_direction   3   unit vector from EE to goal
        yaw_error        1   signed yaw difference (wrapped)
        collision        1   1.0 if self-collision, else 0.0
        last_action    4|6   previous action (zeros at reset)
        ============== ===== ===================================
        → total = 29 (cartesian) or 31 (joint)

    Reward:
        Dense exponential shaping based on proximity and yaw alignment.
        ``0.6 * exp(-5·dist) + 0.4 * exp(-1·|yaw_err|)``
        with time penalty, collision penalty, action-rate penalty,
        and a +10 bonus when both position and yaw thresholds are met.
        One goal per episode, terminates on success or time-out.

    Parameters
    ----------
    robot : str
        Robot configuration key (see :data:`ROBOT_CONFIGS`).
    model_path : str | None
        Override MJCF path (defaults to the config entry).
    time_limit : int
        Max steps per episode (0 = unlimited).
    action_mode : str
        ``"cartesian"`` (4-D IK) or ``"joint"`` (6-D offsets).
    ee_step : float
        EE position step size per action unit (metres). Cartesian only.
    yaw_step : float
        EE yaw step size per action unit (radians). Cartesian only.
    joint_action_scale : float
        Scaling factor for joint-position offsets. Joint mode only.
    reach_threshold : float
        Distance threshold to count as "reached" (metres).
    yaw_threshold : float
        Yaw error threshold to count as "reached" (radians).
    obs_noise : float
        Uniform observation noise amplitude (applied to proprioception).
    action_rate_weight : float
        Penalty weight for ``||a_t - a_{t-1}||²``.
    render_size : tuple[int, int]
        Width × height of the offscreen renderer.
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        robot: str = "ur5e",
        model_path: Optional[str] = None,
        time_limit: int = 1000,
        action_mode: str = "cartesian",
        ee_step: float = 0.06,
        yaw_step: float = 0.5,
        joint_action_scale: float = 0.5,
        reach_threshold: float = 0.06,
        yaw_threshold: float = 0.35,
        obs_noise: float = 0.01,
        action_rate_weight: float = 0.005,
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

        # Action mode: "cartesian" (4-D IK) or "joint" (6-D offsets)
        if action_mode not in ("cartesian", "joint"):
            raise ValueError(f"action_mode must be 'cartesian' or 'joint', got '{action_mode}'")
        self.action_mode = action_mode

        self.time_limit = time_limit
        self.ee_step = ee_step
        self.yaw_step = yaw_step
        self.joint_action_scale = joint_action_scale
        self.reach_threshold = reach_threshold
        self.yaw_threshold = yaw_threshold
        self.obs_noise = obs_noise
        self.action_rate_weight = action_rate_weight
        self.render_size = render_size

        # Physics / control tuning
        self.max_joint_vel = 4.0
        self.ik_damping = 0.02
        self.hold_eps = 0.05
        self.n_substeps = 5
        self.settle_steps = 300

        # Robot-specific home pose and workspace bounds
        self.init_q = cfg.init_q.copy()
        self.goal_bounds = cfg.goal_bounds.copy()
        self.ee_bounds = cfg.ee_bounds.copy()

        # ---- Build MuJoCo model ----
        robot_xml = load_robot_xml(self.model_path)
        self.model_xml = build_reach_xml(robot_xml, render_size, reach_threshold)
        self.model = mujoco.MjModel.from_xml_string(self.model_xml)
        self.data = mujoco.MjData(self.model)

        # Solver settings
        self.model.opt.timestep = 0.002
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.iterations = max(self.model.opt.iterations, 50)
        self.model.opt.gravity[:] = [0.0, 0.0, -9.81]

        # Renderers — three cameras for side-by-side video
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

        # Increase passive damping for stability
        for dof in self.robot_dofs:
            self.model.dof_damping[dof] = max(self.model.dof_damping[dof], 8.0)
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
        self.goal_yaw = 0.0
        self.step_id = 0
        self._init_dist = 1.0
        self._init_yaw_err = math.pi
        self._goals_reached = 0
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)

    # -------------------------------------------------------------- Properties
    @property
    def action_dim(self) -> int:
        """Action dimensionality: 4 (cartesian) or 6 (joint)."""
        return 4 if self.action_mode == "cartesian" else 6

    @property
    def observation_dim(self) -> int:
        """Observation dimensionality: 25 base + action_dim (last_action).

        6 joint_pos + 6 joint_vel + 3 ee_pos + 1 ee_yaw
        + 3 goal_pos + 1 goal_yaw + 3 goal_direction
        + 1 yaw_error + 1 collision + action_dim last_action
        = 29 (cartesian) or 31 (joint)
        """
        return 25 + self.action_dim

    # -------------------------------------------------------------- Reset
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment and return the initial observation."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.step_id = 0
        self._goals_reached = 0
        self._last_targets = self.init_q.copy()
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)

        # Set robot to home pose
        for qi, j in enumerate(self.robot_joints):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            self.data.qpos[self.model.jnt_qposadr[jid]] = self.init_q[qi]
            self.data.qvel[self.model.jnt_dofadr[jid]] = 0.0
            act_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_motor"
            )
            if act_id != -1:
                self.data.ctrl[act_id] = self.init_q[qi]

        mujoco.mj_forward(self.model, self.data)

        # Sample goal — must be reachable and not too close to current EE
        self.goal_pos = self._sample_goal()
        self.goal_yaw = float(self._rng.uniform(-math.pi, math.pi))
        self._place_goal_marker(self.goal_pos, self.goal_yaw)

        for _ in range(self.settle_steps):
            mujoco.mj_step(self.model, self.data)

        self._init_dist = self._ee_goal_dist()
        self._init_yaw_err = abs(self._yaw_error())
        return self._observe()

    def _sample_goal(self) -> np.ndarray:
        """Sample a random reachable goal position.

        The goal must satisfy all constraints from the robot config:
        - Within ``goal_bounds`` box
        - Between ``goal_distance`` (min, max) from the robot base
        - Above ``goal_min_height`` (prevents folding onto table)
        - At least ``goal_min_ee_dist`` from the current EE position
        """
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
            # Too low — arm can just collapse onto it
            if goal[2] < min_height:
                continue
            # Too close / too far from base
            dist_from_base = np.linalg.norm(goal - self._BASE_POS)
            if dist_from_base < min_base or dist_from_base > max_base:
                continue
            # Too close to current EE — trivial reach
            if np.linalg.norm(goal - ee_pos) < min_ee:
                continue
            return goal
        # Fallback — safe position in front of the robot
        return self._BASE_POS + np.array([0.25, 0.0, 0.30])

    def _place_goal_marker(self, pos: np.ndarray, yaw: float = 0.0) -> None:
        """Move and rotate the goal arrow to the desired pose.

        The arrow body’s +X axis is rotated to point along ``yaw``
        (rotation about world Z).
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_body")
        self.model.body_pos[body_id] = pos
        # Quaternion for rotation about Z by `yaw`:
        #   quat = [cos(yaw/2), 0, 0, sin(yaw/2)]
        half = yaw / 2.0
        self.model.body_quat[body_id] = [
            math.cos(half), 0.0, 0.0, math.sin(half)
        ]
        mujoco.mj_forward(self.model, self.data)

    # -------------------------------------------------------------- IK + control
    def _ee_yaw(self) -> float:
        return self._ik.ee_yaw()

    def _desired_ee(self, delta_xyz: np.ndarray, dyaw: float) -> Tuple[np.ndarray, float]:
        pos = self.data.site_xpos[self.ee_site].copy() + delta_xyz
        for i in range(3):
            pos[i] = float(np.clip(pos[i], self.ee_bounds[i, 0], self.ee_bounds[i, 1]))
        yaw = self._ee_yaw() + dyaw
        return pos, yaw

    def _ik_cartesian(self, target_pos: np.ndarray, target_yaw: float) -> np.ndarray:
        return self._ik.solve(target_pos, target_yaw)

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

    def _yaw_error(self) -> float:
        """Signed yaw error wrapped to [-π, π]."""
        err = self.goal_yaw - self._ee_yaw()
        return float((err + math.pi) % (2 * math.pi) - math.pi)

    def _observe(self) -> np.ndarray:
        parts: List[np.ndarray] = []

        # Joint positions — relative to home pose (Isaac Lab style)
        for qi, j in enumerate(self.robot_joints):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            qpos = self.data.qpos[self.model.jnt_qposadr[jid]]
            parts.append(np.array([qpos - self.init_q[qi]]))

        # Joint velocities
        for j in self.robot_joints:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            parts.append(np.array([self.data.qvel[self.model.jnt_dofadr[jid]]]))

        parts.append(self.data.site_xpos[self.ee_site].copy())
        parts.append(np.array([self._ee_yaw()]))
        parts.append(self.goal_pos.copy())
        parts.append(np.array([self.goal_yaw]))

        # Direction from EE to goal (normalised)
        ee2goal = self.goal_pos - self.data.site_xpos[self.ee_site]
        d = np.linalg.norm(ee2goal)
        direction = ee2goal / max(d, 1e-6)
        parts.append(direction)

        # Signed yaw error (wrapped to [-π, π])
        parts.append(np.array([self._yaw_error()]))
        parts.append(np.array([float(self._self_collision_count > 0)]))

        # Last action (Isaac Lab style — helps policy learn smooth control)
        parts.append(self._last_action.copy())

        obs = np.concatenate(parts).astype(np.float32)

        # Observation noise on proprioceptive channels (first 12: joints)
        if self.obs_noise > 0.0:
            noise = self._rng.uniform(
                -self.obs_noise, self.obs_noise, size=12
            ).astype(np.float32)
            obs[:12] += noise

        return obs

    # -------------------------------------------------------------- Reward
    def _compute_reward(self) -> Tuple[float, bool, Dict]:
        dist = self._ee_goal_dist()
        yaw_err = abs(self._yaw_error())    # [0, π]

        pos_ok = dist < self.reach_threshold
        yaw_ok = yaw_err < self.yaw_threshold
        reached = pos_ok and yaw_ok

        # ---- Dense reward: exponential closeness (no init-dist dependency) ----
        #
        # Position: exp(-α·dist)   → 1.0 at goal, ~0.22 at 0.3m, ~0.05 at 0.6m
        # Yaw:      exp(-β·|err|)  → 1.0 at match, ~0.37 at 1 rad
        #
        # This gives a smooth, consistent gradient regardless of episode-specific
        # initial distances, and creates a strong "pull" near the goal.
        pos_reward = math.exp(-5.0 * dist)           # range (0, 1]
        yaw_reward = math.exp(-1.0 * yaw_err)        # range (0, 1]

        # Weight position higher until close, then yaw matters more
        reward = 0.6 * pos_reward + 0.4 * yaw_reward

        # Small time penalty to discourage dawdling
        reward -= 0.01

        # Action rate penalty (Isaac Lab style — smooth motions)
        action_delta = self._last_action - self._prev_action
        reward -= self.action_rate_weight * float(np.dot(action_delta, action_delta))

        # Self-collision penalty
        if self._self_collision_count > 0:
            reward -= 1.0

        # Reach bonus (position + yaw)
        if reached:
            reward += 10.0
            self._goals_reached += 1

        time_up = self.time_limit > 0 and self.step_id >= self.time_limit
        done = reached or time_up

        info = {
            "dist": dist,
            "yaw_err": yaw_err,
            "reached": reached,
            "pos_ok": pos_ok,
            "yaw_ok": yaw_ok,
            "goals_reached": self._goals_reached,
            "self_collisions": self._self_collision_count,
            "ee_pos": self.data.site_xpos[self.ee_site].copy(),
            "ee_yaw": self._ee_yaw(),
            "goal_pos": self.goal_pos.copy(),
            "goal_yaw": self.goal_yaw,
            "is_success": reached,
        }
        return float(reward), done, info

    # -------------------------------------------------------------- Step
    def step(self, action: Iterable[float]) -> StepResult:
        """Execute one environment step.

        Parameters
        ----------
        action : array-like, shape (4,) or (6,)
            Cartesian mode: ``[dx, dy, dz, dyaw]`` each in [-1, 1].
            Joint mode: 6 joint-position offsets each in [-1, 1].

        Returns
        -------
        StepResult
            Named result with ``obs``, ``reward``, ``done``, ``info``.
        """
        act = np.asarray(action, dtype=float).flatten()
        if act.shape[0] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {act.shape[0]}")
        act = np.clip(act, -1.0, 1.0)

        # Track actions for action-rate penalty
        self._prev_action = self._last_action.copy()
        self._last_action = act.astype(np.float32).copy()

        prev_targets = self._last_targets.copy()

        if self.action_mode == "cartesian":
            # --- Cartesian IK path (original) ---
            delta_pos = act[:3] * self.ee_step
            dyaw = act[3] * self.yaw_step

            target_pos, target_yaw = self._desired_ee(delta_pos, dyaw)
            qvel_cmd = self._ik_cartesian(target_pos, target_yaw)
            qvel_cmd = np.clip(qvel_cmd, -self.max_joint_vel, self.max_joint_vel)

            ik_gain = 0.15
            if np.linalg.norm(act) < self.hold_eps:
                qpos_targets = self._last_targets.copy()
            else:
                qpos_targets = self._last_targets + qvel_cmd * ik_gain
        else:
            # --- Joint-space offsets (Isaac Lab style) ---
            qpos_targets = self._last_targets + act * self.joint_action_scale

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
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.model, self.data)

        reward, done, info = self._compute_reward()
        self.step_id += 1
        return StepResult(self._observe(), reward, done, info)

    # -------------------------------------------------------------- Render
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the scene.

        Parameters
        ----------
        mode : str
            ``"rgb_array"`` returns a side-by-side NumPy frame.
            ``"human"`` returns ``None`` (viewer handles display).
        """
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
