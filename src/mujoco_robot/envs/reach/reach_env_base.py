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

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Optional, Tuple

import gymnasium
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mujoco_robot.robots.actuators import (
    configure_position_actuators,
    resolve_robot_actuators,
)
from mujoco_robot.robots.configs import get_robot_config
from mujoco_robot.core.ik_controller import (
    IKController,
    orientation_error_axis_angle,
    quat_error_magnitude,
    quat_multiply,
    quat_unique,
)
from mujoco_robot.core.collision import CollisionDetector
from mujoco_robot.core.xml_builder import load_robot_xml, build_reach_xml
from mujoco_robot.envs.manager_based import ManagerRuntime
from mujoco_robot.tasks.manager_based.manipulation.reach.mdp import (
    ActionManager,
    CommandManager,
    ReachMDPCfg,
    ReachRewardCfg,
    ObservationManager,
    RewardManager,
    TerminationManager,
    make_default_reach_mdp_cfg,
)


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
        time_limit: int = 360,
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

    Reward (Isaac Lab reach manager cfg):
        ``-0.2 * position_error`` + ``0.1 * position_error_tanh(std=0.1)`` +
        ``-0.1 * orientation_error`` + ``-w_ar * ||Δa||²`` + ``-w_jv * ||q̇||²``
        where ``w_ar`` and ``w_jv`` follow the Isaac Lab curriculum schedule.
    """

    # Subclasses may refine these; defaults assume 6-DOF action.
    _action_dim: int = 6
    DEFAULT_TIME_LIMIT: int = 360

    def __init__(
        self,
        robot: str = "ur5e",
        model_path: Optional[str] = None,
        time_limit: int = DEFAULT_TIME_LIMIT,
        ee_step: float = 0.06,
        ori_step: float = 0.5,
        ori_abs_max: float = np.pi,
        joint_action_scale: float = 0.5,
        reach_threshold: float = 0.001,  # meters
        ori_threshold: float = 0.01,  # radians
        terminate_on_success: bool = False,
        terminate_on_collision: bool = False,
        goal_resample_time_range_s: Tuple[float, float] | None = None,
        success_hold_steps: int = 10,
        success_bonus: float = 0.25,
        stay_reward_weight: float = 0.05,
        resample_on_success: bool = False,
        obs_noise: float = 0.01,
        action_rate_weight: float = 0.0001,
        joint_vel_weight: float = 0.0001,
        randomize_init: bool = True,
        init_q_range: Tuple[float, float] = (0.75, 1.25),
        actuator_kp: float = 100.0,
        min_joint_damping: float = 20.0,
        min_joint_frictionloss: float = 1.0,
        render_size: Tuple[int, int] = (960, 720),
        seed: Optional[int] = None,
        reward_cfg: ReachRewardCfg | None = None,
        mdp_cfg: ReachMDPCfg | None = None,
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
        self.action_rate_weight = float(action_rate_weight)
        self.joint_vel_weight = joint_vel_weight
        self.randomize_init = randomize_init
        self.init_q_range = init_q_range
        self.actuator_kp = float(actuator_kp)
        self.min_joint_damping = float(min_joint_damping)
        self.min_joint_frictionloss = float(min_joint_frictionloss)
        self.render_size = render_size
        self._reward_cfg = reward_cfg

        # Isaac-style: command resampling based on elapsed wall-clock time.
        self.reach_threshold = reach_threshold
        self.ori_threshold = ori_threshold
        self.terminate_on_success = terminate_on_success
        self.terminate_on_collision = terminate_on_collision
        self.success_hold_steps = max(1, int(success_hold_steps))
        self.success_bonus = float(success_bonus)
        self.stay_reward_weight = float(stay_reward_weight)
        self.resample_on_success = bool(resample_on_success)
        default_step_dt = (1.0 / 60.0) * 2.0
        if goal_resample_time_range_s is None:
            # Default: avoid in-episode goal switches when episode is finite.
            if self.time_limit > 0:
                episode_s = float(self.time_limit) * default_step_dt
                no_resample_s = episode_s + default_step_dt
                self.goal_resample_time_range_s = (no_resample_s, no_resample_s)
            else:
                # Infinite-horizon fallback keeps periodic commands.
                self.goal_resample_time_range_s = (4.0, 4.0)
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
        self.hold_eps = 0.0
        self.n_substeps = 2
        self.settle_steps = 300

        # Robot-specific home pose and workspace bounds
        self.init_q = cfg.init_q.copy()
        self.goal_bounds = cfg.goal_bounds.copy()
        self.ee_bounds = cfg.ee_bounds.copy()
        self._table_spawn_margin_xy = 0.04
        self._table_goal_z_margin = 0.08
        self._table_xy_bounds: np.ndarray | None = None
        self._table_top_z: float | None = None

        # ---- Build MuJoCo model ----
        robot_xml = load_robot_xml(self.model_path)
        marker_size = 0.06
        self.model_xml = build_reach_xml(robot_xml, render_size, marker_size)
        self.model = mujoco.MjModel.from_xml_string(self.model_xml)
        self.data = mujoco.MjData(self.model)

        # Solver settings
        self.model.opt.timestep = 1.0 / 60.0
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
        self._refresh_table_bounds()

        actuator_handles = resolve_robot_actuators(self.model, robot)
        self.robot_joints = list(actuator_handles.joint_names)
        self.robot_actuators = list(actuator_handles.actuator_ids)
        self.robot_dofs = list(actuator_handles.dof_ids)
        self._robot_joint_ids = list(actuator_handles.joint_ids)
        self._robot_qpos_ids = list(actuator_handles.qpos_ids)

        # Tune position-servo stiffness and passive damping for stability.
        # In MuJoCo, very stiff servo gains can ring around small targets,
        # so we use a softer kp and stronger passive damping by default.
        configure_position_actuators(
            self.model,
            actuator_handles,
            min_damping=self.min_joint_damping,
            min_frictionloss=self.min_joint_frictionloss,
            kp=self.actuator_kp,
        )

        # --- Self-collision detection ---
        self._collision_detector = CollisionDetector(self.model)
        self._self_collision_count = 0

        # --- IK controller ---
        self._ik = IKController(
            self.model, self.data, self.ee_site, self.robot_dofs,
            damping=self.ik_damping,
        )

        # State
        self._last_targets = self.init_q.copy()
        self.goal_pos = np.zeros(3)
        self.goal_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._home_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.step_id = 0
        self._init_dist = 1.0
        self._init_ori_err = np.pi
        self._goals_reached = 0
        self._goals_held = 0
        self._goals_resampled = 0
        self._prev_success = False
        self._success_streak_steps = 0
        self._goal_hold_completed = False
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)

        # Manager-based MDP composition (IsaacLab-style).
        self._mdp_cfg = self._resolve_mdp_cfg(mdp_cfg)
        if self._mdp_cfg.command_term.resampling_time_range_s != self.goal_resample_time_range_s:
            self._mdp_cfg.command_term = replace(
                self._mdp_cfg.command_term,
                resampling_time_range_s=self.goal_resample_time_range_s,
            )
        self._manager_runtime = ManagerRuntime()
        self._manager_runtime.add_many(
            action=ActionManager(self, self._mdp_cfg.action_term),
            command=CommandManager(self, self._mdp_cfg.command_term),
            observation=ObservationManager(self, self._mdp_cfg.observation_terms),
            reward=RewardManager(self, self._mdp_cfg.reward_terms),
            termination=TerminationManager(
                self,
                self._mdp_cfg.success_term,
                self._mdp_cfg.failure_term,
                self._mdp_cfg.timeout_term,
            ),
        )
        # Backward-compatible aliases for existing internal accesses.
        self._action_manager = self._manager("action")
        self._command_manager = self._manager("command")
        self._observation_manager = self._manager("observation")
        self._reward_manager = self._manager("reward")
        self._termination_manager = self._manager("termination")

    # -------------------------------------------------------------- Properties
    def _build_default_mdp_cfg(self) -> ReachMDPCfg:
        """Return default manager-based MDP config for this environment."""
        return make_default_reach_mdp_cfg(reward_cfg=self._reward_cfg)

    def _resolve_mdp_cfg(self, mdp_cfg: ReachMDPCfg | None) -> ReachMDPCfg:
        cfg = mdp_cfg or self._build_default_mdp_cfg()
        defaults = self._build_default_mdp_cfg()
        if cfg.action_term is None:
            cfg.action_term = defaults.action_term
        if cfg.command_term is None:
            cfg.command_term = defaults.command_term
        if not cfg.observation_terms:
            cfg.observation_terms = defaults.observation_terms
        if not cfg.reward_terms:
            cfg.reward_terms = defaults.reward_terms
        if cfg.success_term is None:
            cfg.success_term = defaults.success_term
        if cfg.failure_term is None:
            cfg.failure_term = defaults.failure_term
        if cfg.timeout_term is None:
            cfg.timeout_term = defaults.timeout_term
        return cfg

    def _manager(self, name: str):
        """Return one manager from the runtime registry."""
        return self._manager_runtime.require(name)

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
        if hasattr(self, "_manager_runtime") and self._manager_runtime.has("observation"):
            return int(self._manager("observation").dim)
        return 19 + self.action_dim

    # -------------------------------------------------------------- Reset
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment and return the initial observation."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.step_id = 0
        self._goals_reached = 0
        self._goals_held = 0
        self._goals_resampled = 0
        self._prev_success = False
        self._success_streak_steps = 0
        self._goal_hold_completed = False
        self._self_collision_count = 0
        self._last_targets = self.init_q.copy()
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)

        # Set robot to home pose (with optional Isaac Lab-style randomization)
        self._total_episodes += 1
        for qi, (jid, qpos_adr, dof_adr, act_id) in enumerate(
            zip(
                self._robot_joint_ids,
                self._robot_qpos_ids,
                self.robot_dofs,
                self.robot_actuators,
            )
        ):
            q_home = self.init_q[qi]
            if self.randomize_init:
                scale = float(self._rng.uniform(*self.init_q_range))
                q_home = q_home * scale
                lo, hi = self.model.jnt_range[jid]
                if lo < hi:
                    q_home = float(np.clip(q_home, lo, hi))
            self.data.qpos[qpos_adr] = q_home
            self.data.qvel[dof_adr] = 0.0
            self.data.ctrl[act_id] = q_home
        # Update last targets to match randomized start
        for qi, qpos_adr in enumerate(self._robot_qpos_ids):
            self._last_targets[qi] = self.data.qpos[qpos_adr]

        mujoco.mj_forward(self.model, self.data)

        # Store the home orientation for orientation curriculum
        self._home_quat = self._ee_quat()

        # Sample initial command through command manager.
        self._manager("command").reset()
        self._resample_goal()

        for _ in range(self.settle_steps):
            mujoco.mj_step(self.model, self.data)

        self._init_dist = self._ee_goal_dist()
        self._init_ori_err = self._orientation_error_magnitude()
        return self._observe()

    # -------------------------------------------------------------- Goal sampling
    def _refresh_table_bounds(self) -> None:
        """Cache table XY bounds / top Z if a table geom is present."""
        self._table_xy_bounds = None
        self._table_top_z = None

        table_gid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "table"
        )
        if table_gid < 0:
            return
        if int(self.model.geom_type[table_gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            return

        pos = self.model.geom_pos[table_gid]
        size = self.model.geom_size[table_gid]
        self._table_xy_bounds = np.array([
            [pos[0] - size[0], pos[0] + size[0]],
            [pos[1] - size[1], pos[1] + size[1]],
        ], dtype=float)
        self._table_top_z = float(pos[2] + size[2])

    def _goal_sampling_bounds(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Goal XYZ sampling bounds, clipped to table footprint when available."""
        x_lo, x_hi = float(self.goal_bounds[0, 0]), float(self.goal_bounds[0, 1])
        y_lo, y_hi = float(self.goal_bounds[1, 0]), float(self.goal_bounds[1, 1])
        z_lo, z_hi = float(self.goal_bounds[2, 0]), float(self.goal_bounds[2, 1])

        if self._table_xy_bounds is not None:
            x_lo = max(x_lo, float(self._table_xy_bounds[0, 0]) + self._table_spawn_margin_xy)
            x_hi = min(x_hi, float(self._table_xy_bounds[0, 1]) - self._table_spawn_margin_xy)
            y_lo = max(y_lo, float(self._table_xy_bounds[1, 0]) + self._table_spawn_margin_xy)
            y_hi = min(y_hi, float(self._table_xy_bounds[1, 1]) - self._table_spawn_margin_xy)
        if self._table_top_z is not None:
            z_lo = max(z_lo, self._table_top_z + self._table_goal_z_margin)

        # Keep ranges valid even for unusual custom robot/table setups.
        if x_lo >= x_hi:
            if self._table_xy_bounds is not None:
                x_lo = float(self._table_xy_bounds[0, 0]) + 0.01
                x_hi = float(self._table_xy_bounds[0, 1]) - 0.01
            if x_lo >= x_hi:
                x_lo, x_hi = float(self.goal_bounds[0, 0]), float(self.goal_bounds[0, 1])
        if y_lo >= y_hi:
            if self._table_xy_bounds is not None:
                y_lo = float(self._table_xy_bounds[1, 0]) + 0.01
                y_hi = float(self._table_xy_bounds[1, 1]) - 0.01
            if y_lo >= y_hi:
                y_lo, y_hi = float(self.goal_bounds[1, 0]), float(self.goal_bounds[1, 1])
        if z_lo >= z_hi:
            z_lo, z_hi = float(self.goal_bounds[2, 0]), float(self.goal_bounds[2, 1])
            if z_lo >= z_hi:
                z_mid = 0.5 * (z_lo + z_hi)
                z_lo, z_hi = z_mid - 0.01, z_mid + 0.01

        return (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi)

    def _sample_goal(self) -> np.ndarray:
        """Sample a random reachable goal position."""
        cfg = get_robot_config(self.robot)
        ee_pos = self.data.site_xpos[self.ee_site].copy()
        min_base, max_base = cfg.goal_distance
        min_height = cfg.goal_min_height
        min_ee = cfg.goal_min_ee_dist
        (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = self._goal_sampling_bounds()
        if self._table_top_z is not None:
            min_height = max(min_height, self._table_top_z + self._table_goal_z_margin)

        for _ in range(500):
            goal = np.array([
                self._rng.uniform(x_lo, x_hi),
                self._rng.uniform(y_lo, y_hi),
                self._rng.uniform(z_lo, z_hi),
            ])
            if goal[2] < min_height:
                continue
            dist_from_base = np.linalg.norm(goal - self._BASE_POS)
            if dist_from_base < min_base or dist_from_base > max_base:
                continue
            if np.linalg.norm(goal - ee_pos) < min_ee:
                continue
            return goal
        fallback = self._BASE_POS + np.array([0.25, 0.0, 0.30])
        fallback[0] = float(np.clip(fallback[0], x_lo, x_hi))
        fallback[1] = float(np.clip(fallback[1], y_lo, y_hi))
        fallback[2] = float(np.clip(fallback[2], z_lo, z_hi))
        return fallback

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
        for k, jid in enumerate(self._robot_joint_ids):
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
        """Build observation vector from configured observation terms."""
        obs = self._manager("observation").observe()

        # Observation noise on proprioceptive channels (first 12: joint_pos + joint_vel)
        # Matches Isaac Lab: Unoise(n_min=-0.01, n_max=0.01) on both terms.
        if self.obs_noise > 0.0 and obs.shape[0] >= 12:
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
        """Apply the currently sampled command pose as the active goal."""
        pose_command = self._manager("command").pose_command
        self.goal_pos = pose_command[:3].astype(float)
        self.goal_quat = quat_unique(pose_command[3:7].astype(float))
        self._place_goal_marker(self.goal_pos, self.goal_quat)
        self._success_streak_steps = 0
        self._goal_hold_completed = False

    def _maybe_resample_goal(self) -> tuple[bool, str]:
        """Advance command timer and resample when the sampled duration elapses."""
        step_dt = self.model.opt.timestep * self.n_substeps
        if not self._manager("command").step(step_dt):
            return False, "none"

        self._goals_resampled += 1
        self._resample_goal()
        return True, "timer"

    def _resample_goal_after_success(self) -> tuple[bool, str]:
        """Resample command pose immediately after hold-success."""
        self._manager("command").reset()
        self._goals_resampled += 1
        self._resample_goal()
        return True, "success"

    def resample_goal_now(self) -> np.ndarray:
        """Force a new goal command immediately.

        This is intended for interactive evaluation/debug workflows.
        Returns the newly active goal position.
        """
        self._manager("command").reset()
        self._goals_resampled += 1
        self._resample_goal()
        # Treat next step as a fresh goal attempt.
        self._prev_success = False
        return self.goal_pos.copy()

    def _compute_done_flags(
        self,
        dist: float,
        ori_err_mag: float,
    ) -> tuple[bool, bool, bool, bool, bool]:
        """Compute done flags via the configured termination manager."""
        ctx = {"dist": float(dist), "ori_err": float(ori_err_mag)}
        flags = self._manager("termination").compute(ctx)
        return (
            bool(flags["success"]),
            bool(flags["failure"]),
            bool(flags["terminated"]),
            bool(flags["time_out"]),
            bool(flags["done"]),
        )

    def _compute_reward(self) -> Tuple[float, bool, Dict]:
        """Compute reward and done flags from configured manager terms."""
        # Keep goal snapshots so returned metrics stay consistent even if we
        # resample the command pose at the end of this step.
        goal_pos_step = self.goal_pos.copy()
        goal_quat_step = self.goal_quat.copy()

        dist = self._ee_goal_dist()
        ori_err_mag = self._orientation_error_magnitude()
        action_delta = self._last_action - self._prev_action
        action_rate_l2 = float(np.dot(action_delta, action_delta))
        joint_vel = self.data.qvel[self.robot_dofs]
        joint_vel_l2 = float(np.dot(joint_vel, joint_vel))
        ctx = {
            "dist": dist,
            "ori_err": ori_err_mag,
            "action_rate_l2": action_rate_l2,
            "joint_vel_l2": joint_vel_l2,
        }

        reward, raw_reward_terms, _weighted_terms = self._manager("reward").compute(ctx)

        success, failure, terminated, time_up, done = self._compute_done_flags(dist, ori_err_mag)
        if success and not self._prev_success:
            self._goals_reached += 1
        self._prev_success = success
        if success:
            self._success_streak_steps += 1
        else:
            self._success_streak_steps = 0

        hold_success = bool(success and self._success_streak_steps >= self.success_hold_steps)
        first_hold = bool(hold_success and not self._goal_hold_completed)
        if first_hold:
            self._goal_hold_completed = True
            self._goals_held += 1

        step_dt = float(self.model.opt.timestep * self.n_substeps)
        stay_reward = float(self.stay_reward_weight * step_dt if hold_success else 0.0)
        success_bonus = float(self.success_bonus if first_hold else 0.0)
        reward = float(reward + stay_reward + success_bonus)
        success_streak_steps = int(self._success_streak_steps)

        goal_resampled = False
        goal_resample_reason = "none"
        if not done:
            if self.resample_on_success and first_hold:
                goal_resampled, goal_resample_reason = self._resample_goal_after_success()
            else:
                goal_resampled, goal_resample_reason = self._maybe_resample_goal()
        command_manager = self._manager("command")

        info = {
            "dist": dist,
            "ori_err": ori_err_mag,
            "within_position_threshold": bool(dist < self.reach_threshold),
            "within_orientation_threshold": bool(ori_err_mag < self.ori_threshold),
            "success": success,
            "hold_success": hold_success,
            "first_hold_success": first_hold,
            "success_streak_steps": success_streak_steps,
            "success_hold_steps": self.success_hold_steps,
            "success_bonus": success_bonus,
            "stay_reward": stay_reward,
            "failure": failure,
            "terminated": terminated,
            "time_out": time_up,
            "goal_resample_elapsed_s": command_manager.elapsed_s,
            "goal_resample_target_s": command_manager.target_s,
            "goal_resampled": goal_resampled,
            "goal_resample_reason": goal_resample_reason,
            "goals_reached": self._goals_reached,
            "goals_held": self._goals_held,
            "goals_resampled": self._goals_resampled,
            "self_collisions": self._self_collision_count,
            "ee_pos": self.data.site_xpos[self.ee_site].copy(),
            "ee_quat": self._ee_quat(),
            # Goal used to compute this step's dist/orientation metrics.
            "goal_pos": goal_pos_step,
            "goal_quat": goal_quat_step,
            # Active command after potential end-of-step resample.
            "active_goal_pos": self.goal_pos.copy(),
            "active_goal_quat": self.goal_quat.copy(),
        }
        if self._mdp_cfg.include_reward_terms_in_info:
            reward_terms = dict(raw_reward_terms)
            reward_terms["success_bonus"] = success_bonus
            reward_terms["stay_reward"] = stay_reward
            reward_terms["step_dt"] = step_dt
            info["reward_terms"] = reward_terms
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

        # Delegate to configured action term via manager
        qpos_targets = self._manager("action").compute_joint_targets(act)
        qpos_targets = self._clamp_to_limits(qpos_targets)
        self._last_targets = qpos_targets.copy()

        for k, act_id in enumerate(self.robot_actuators):
            self.data.ctrl[act_id] = qpos_targets[k]

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        # Detect self-collision
        self._self_collision_count = self._collision_detector.count(self.data)

        reward, done, info = self._compute_reward()
        self.step_id += 1
        return StepResult(self._observe(), reward, done, info)

    # -------------------------------------------------------------- Render
    def _draw_metrics_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw a translucent HUD with live metrics on the rendered frame.

        Metrics displayed:
        - Distance to target (m)
        - Orientation error (deg)
        - Per-joint angles (deg) and velocities (deg/s)
        - Step counter, goals reached, self-collisions
        """
        h, w = frame.shape[:2]

        # --- Gather metrics ---
        dist = self._ee_goal_dist()
        ori_err_rad = self._orientation_error_magnitude()
        ori_err_deg = float(np.degrees(ori_err_rad))

        joint_pos_rad = np.array(
            [self.data.qpos[qid] for qid in self._robot_qpos_ids]
        )
        joint_vel_rad = self.data.qvel[self.robot_dofs]
        joint_pos_deg = np.degrees(joint_pos_rad)
        joint_vel_deg = np.degrees(joint_vel_rad)

        ee_pos = self.data.site_xpos[self.ee_site].copy()

        # --- Build text lines ---
        lines: list[tuple[str, tuple[int, int, int]]] = []
        # Header
        lines.append((f"Step {self.step_id:>5d} / {self.time_limit}", (255, 255, 255)))
        lines.append(("", (255, 255, 255)))  # spacer

        # Distance: green when close, yellow mid, red far
        if dist < self.reach_threshold:
            dist_color = (0, 255, 0)
        elif dist < self.reach_threshold * 5:
            dist_color = (255, 255, 0)
        else:
            dist_color = (255, 100, 100)
        lines.append((f"Distance:    {dist*100:6.2f} cm  ({dist:.4f} m)", dist_color))

        # Orientation error: green < threshold, yellow mid, red far
        if ori_err_rad < self.ori_threshold:
            ori_color = (0, 255, 0)
        elif ori_err_deg < 30.0:
            ori_color = (255, 255, 0)
        else:
            ori_color = (255, 100, 100)
        lines.append((f"Ori error:   {ori_err_deg:6.2f} deg ({ori_err_rad:.4f} rad)", ori_color))
        lines.append(("", (255, 255, 255)))  # spacer

        # EE position
        lines.append((
            f"EE pos:  x={ee_pos[0]:+.3f}  y={ee_pos[1]:+.3f}  z={ee_pos[2]:+.3f} m",
            (200, 200, 255),
        ))
        lines.append((
            f"Goal:    x={self.goal_pos[0]:+.3f}  y={self.goal_pos[1]:+.3f}  z={self.goal_pos[2]:+.3f} m",
            (200, 255, 200),
        ))
        lines.append(("", (255, 255, 255)))  # spacer

        # Joint statistics
        lines.append(("Joint        Angle (deg)  Vel (deg/s)", (180, 180, 180)))
        lines.append(("-" * 40, (100, 100, 100)))
        for j in range(len(self.robot_joints)):
            jname = self.robot_joints[j]
            # Truncate long joint names for display
            short = jname[-12:] if len(jname) > 12 else jname
            lines.append((
                f"{short:<12s} {joint_pos_deg[j]:+8.2f}     {joint_vel_deg[j]:+8.2f}",
                (220, 220, 220),
            ))
        lines.append(("", (255, 255, 255)))  # spacer

        # Episode stats
        lines.append((f"Goals reached:    {self._goals_reached}", (180, 220, 255)))
        lines.append((f"Goals held:       {self._goals_held}", (180, 220, 255)))
        lines.append((f"Goals resampled:  {self._goals_resampled}", (180, 220, 255)))
        lines.append((f"Self-collisions:  {self._self_collision_count}", (180, 220, 255)))

        # --- Render text onto frame with PIL ---
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img, "RGBA")

        # Use a small monospaced-like font; fall back to default if needed.
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
        except (IOError, OSError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSansMono.ttf", 13)
            except (IOError, OSError):
                font = ImageFont.load_default()

        line_h = 16
        pad = 6
        panel_w = 340
        panel_h = pad * 2 + line_h * len(lines)
        panel_x = w - panel_w - 8
        panel_y = 8

        # Semi-transparent dark background
        draw.rectangle(
            [panel_x, panel_y, panel_x + panel_w, panel_y + panel_h],
            fill=(0, 0, 0, 160),
        )

        y = panel_y + pad
        for text, color in lines:
            if text:
                draw.text((panel_x + pad, y), text, fill=(*color, 255), font=font)
            y += line_h

        return np.array(img)

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the scene with an optional metrics overlay."""
        if mode == "rgb_array":
            self._renderer_top.update_scene(self.data, camera="top")
            frame_top = self._renderer_top.render()
            self._renderer_side.update_scene(self.data, camera="side")
            frame_side = self._renderer_side.render()
            self._renderer_ee.update_scene(self.data, camera="ee_cam")
            frame_ee = self._renderer_ee.render()
            frame = np.concatenate([frame_top, frame_side, frame_ee], axis=1)
            return self._draw_metrics_overlay(frame)
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
