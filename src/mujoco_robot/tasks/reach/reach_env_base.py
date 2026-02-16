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

from dataclasses import replace
from typing import Dict, Iterable, Optional, Tuple

import gymnasium
import mujoco
import numpy as np

from mujoco_robot.assets.configs import get_robot_config
from mujoco_robot.envs import ManagerRuntime
from mujoco_robot.envs.step_result import StepResult
from mujoco_robot.tasks.reach.scene_builder import build_reach_scene
from mujoco_robot.tasks.reach.success_tracker import SuccessTracker
from mujoco_robot.tasks.reach.mdp import (
    ActionManager,
    CommandManager,
    ReachMDPCfg,
    ReachRewardCfg,
    ObservationManager,
    RewardManager,
    TerminationManager,
    make_default_reach_mdp_cfg,
)
from mujoco_robot.tasks.reach.goals import (
    goal_sampling_bounds,
    maybe_resample_goal_by_timer,
    resample_goal_after_success,
    resample_goal_from_command,
    sample_goal_position,
    sample_goal_quaternion,
)
from mujoco_robot.tasks.reach.kinematics import (
    clamp_joint_targets,
    desired_ee_absolute_pose,
    desired_ee_relative_pose,
    ee_goal_distance,
    ee_quaternion,
    ik_cartesian_joint_targets,
    orientation_error_magnitude,
    orientation_error_vector,
)
from mujoco_robot.tasks.reach.rendering import (
    compose_multi_camera_frame,
    draw_metrics_overlay,
)
from mujoco_robot.tasks.reach.resetting import (
    initialize_goal_and_settle,
    initialize_robot_state,
    reset_episode_state,
)
from mujoco_robot.tasks.reach.rewarding import compute_done_flags, compute_step_reward


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
        actions          6   previous action (zeros at reset)
        ============== ===== ===================================
        → total = 25   (matches Isaac Lab reach env exactly)

    Reward (default manager cfg):
        Dense bounded proximity in ``[0, 1]`` with configurable position/orientation
        weighting and optional clipping. Legacy Isaac-style additive terms can be
        enabled via ``ReachRewardCfg(reward_mode="legacy_isaac")``.
    """

    # Subclasses may refine these; defaults assume 6-DOF action.
    _action_dim: int = 6
    DEFAULT_TIME_LIMIT: int = 360

    def __init__(
        self,
        robot: str = "ur3e",
        model_path: Optional[str] = None,
        time_limit: int = DEFAULT_TIME_LIMIT,
        ee_step: float = 0.06,
        ori_step: float = 0.5,
        ori_abs_max: float = np.pi,
        joint_action_scale: float = 0.5,
        reach_threshold: float = 0.01,  # meters
        ori_threshold: float = 0.1,  # radians
        terminate_on_success: bool = False,
        terminate_on_collision: bool = False,
        goal_resample_time_range_s: Tuple[float, float] | None = None,
        goal_roll_range: Tuple[float, float] = (0.0, 0.0),
        goal_pitch_range: Tuple[float, float] = (0.0, 0.0),
        goal_yaw_range: Tuple[float, float] = (-np.pi, np.pi),
        success_hold_steps: int = 10,
        success_bonus: float = 0.25,
        stay_reward_weight: float = 0.05,
        resample_on_success: bool = False,
        obs_noise: float = 0.01,
        action_rate_weight: float = 0.0001,
        joint_vel_weight: float = 0.0001,
        randomize_init: bool = True,
        init_q_range: Tuple[float, float] = (0.75, 1.25),
        actuator_kp: float = 500.0,
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

        def _validate_angle_range(name: str, value: Tuple[float, float]) -> Tuple[float, float]:
            lo_v = float(value[0])
            hi_v = float(value[1])
            if lo_v > hi_v:
                raise ValueError(f"{name} must satisfy min <= max, got {value}")
            return lo_v, hi_v

        self.goal_roll_range = _validate_angle_range("goal_roll_range", goal_roll_range)
        self.goal_pitch_range = _validate_angle_range("goal_pitch_range", goal_pitch_range)
        self.goal_yaw_range = _validate_angle_range("goal_yaw_range", goal_yaw_range)
        # Curriculum targets (Isaac Lab reach defaults):
        # action_rate: -0.0001 -> -0.005 over 4500 steps
        # joint_vel:   -0.0001 -> -0.001 over 4500 steps
        # We keep positive magnitudes here; signs are applied in reward term helpers.
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

        # ---- Build MuJoCo scene (model, renderers, actuators, IK) ----
        build_reach_scene(self)

        # State
        self._last_targets = self.init_q.copy()
        self.goal_pos = np.zeros(3)
        self.goal_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._home_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.step_id = 0
        self._init_dist = 1.0
        self._init_ori_err = np.pi
        self._success_tracker = SuccessTracker(
            hold_steps=self.success_hold_steps,
            bonus=self.success_bonus,
            stay_weight=self.stay_reward_weight,
        )
        self._goals_resampled = 0
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self._last_reward = 0.0
        self._last_step_info: Dict = {}
        self._last_obs: np.ndarray | None = None

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
        """Observation dimensionality: 19 base + action_dim (actions).

        Matches Isaac Lab reach env: joint_pos(6) + joint_vel(6) +
        pose_command(7) + actions(action_dim).
        """
        if hasattr(self, "_manager_runtime") and self._manager_runtime.has("observation"):
            return int(self._manager("observation").dim)
        return 19 + self.action_dim

    # Backward-compatible proxies — delegate to SuccessTracker
    @property
    def _goals_reached(self) -> int:
        return self._success_tracker.goals_reached

    @_goals_reached.setter
    def _goals_reached(self, v: int) -> None:
        self._success_tracker.goals_reached = v

    @property
    def _goals_held(self) -> int:
        return self._success_tracker.goals_held

    @_goals_held.setter
    def _goals_held(self, v: int) -> None:
        self._success_tracker.goals_held = v

    @property
    def _prev_success(self) -> bool:
        return self._success_tracker._prev_success

    @_prev_success.setter
    def _prev_success(self, v: bool) -> None:
        self._success_tracker._prev_success = v

    @property
    def _success_streak_steps(self) -> int:
        return self._success_tracker.streak_steps

    @_success_streak_steps.setter
    def _success_streak_steps(self, v: int) -> None:
        self._success_tracker.streak_steps = v

    @property
    def _goal_hold_completed(self) -> bool:
        return self._success_tracker._hold_completed

    @_goal_hold_completed.setter
    def _goal_hold_completed(self, v: bool) -> None:
        self._success_tracker._hold_completed = v

    # -------------------------------------------------------------- Reset
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment and return the initial observation."""
        reset_episode_state(self, seed)
        initialize_robot_state(self)
        initialize_goal_and_settle(self)
        obs = self._observe()
        self._last_obs = obs.copy()
        return obs

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

    def _center_top_camera_over_table(self) -> None:
        """Recenter the fixed `top` camera over the table footprint."""
        if self._table_xy_bounds is None:
            return
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
        if cam_id < 0:
            return
        table_center = np.array([
            0.5 * (self._table_xy_bounds[0, 0] + self._table_xy_bounds[0, 1]),
            0.5 * (self._table_xy_bounds[1, 0] + self._table_xy_bounds[1, 1]),
        ], dtype=float)
        self.model.cam_pos[cam_id, 0] = table_center[0]
        self.model.cam_pos[cam_id, 1] = table_center[1]

    def _goal_sampling_bounds(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Goal XYZ sampling bounds, clipped to table footprint when available."""
        return goal_sampling_bounds(self)

    def _sample_goal(self) -> np.ndarray:
        """Sample a random reachable goal position."""
        return sample_goal_position(self)

    def _sample_goal_quat(self) -> np.ndarray:
        """Sample a random goal orientation with curriculum."""
        return sample_goal_quaternion(self)

    def _place_goal_marker(self, pos: np.ndarray, quat: np.ndarray) -> None:
        """Move and rotate the goal marker to the desired pose."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_body")
        self.model.body_pos[body_id] = pos
        self.model.body_quat[body_id] = quat
        mujoco.mj_forward(self.model, self.data)

    # -------------------------------------------------------------- IK + control helpers
    def _ee_quat(self) -> np.ndarray:
        """Current EE orientation as unit quaternion (w,x,y,z)."""
        return ee_quaternion(self)

    def _desired_ee_relative(
        self,
        delta_xyz: np.ndarray,
        delta_ori: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute desired EE pose from **relative** Cartesian action."""
        return desired_ee_relative_pose(self, delta_xyz, delta_ori)

    def _desired_ee_absolute(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute desired EE pose from **absolute** Cartesian action.

        Action layout (all in [-1, 1]):
            - ``action[:3]`` maps linearly to workspace XYZ bounds.
            - ``action[3:6]`` is an absolute axis-angle vector scaled by
              ``ori_abs_max`` and applied about ``_home_quat``.
        """
        return desired_ee_absolute_pose(self, action)

    def _ik_cartesian(self, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        return ik_cartesian_joint_targets(self, target_pos, target_quat)

    def _clamp_to_limits(self, targets: np.ndarray) -> np.ndarray:
        return clamp_joint_targets(self, targets)

    # -------------------------------------------------------------- Observation
    def _ee_goal_dist(self) -> float:
        return ee_goal_distance(self)

    def _orientation_error(self) -> np.ndarray:
        """Axis-angle orientation error vector (3-D), from EE toward goal."""
        return orientation_error_vector(self)

    def _orientation_error_magnitude(self) -> float:
        """Scalar orientation error in radians ∈ [0, π]."""
        return orientation_error_magnitude(self)

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
        resample_goal_from_command(self)

    def _maybe_resample_goal(self) -> tuple[bool, str]:
        """Advance command timer and resample when the sampled duration elapses."""
        return maybe_resample_goal_by_timer(self)

    def _resample_goal_after_success(self) -> tuple[bool, str]:
        """Resample command pose immediately after hold-success."""
        return resample_goal_after_success(self)

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
        return compute_done_flags(self, dist, ori_err_mag)

    def _compute_reward(self) -> Tuple[float, bool, Dict]:
        """Compute reward and done flags from configured manager terms."""
        return compute_step_reward(self)

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
        self._last_reward = float(reward)
        self._last_step_info = dict(info)
        obs = self._observe()
        self._last_obs = obs.copy()
        self.step_id += 1
        return StepResult(obs, reward, done, info)

    # -------------------------------------------------------------- Render
    def _draw_metrics_overlay(
        self,
        frame: np.ndarray,
        panel_x: int | None = None,
        panel_y: int = 8,
        panel_w: int | None = None,
    ) -> np.ndarray:
        """Draw a translucent HUD with live metrics on the rendered frame.

        Metrics displayed:
        - Distance to target (m)
        - Orientation error (deg)
        - Current reward
        - Per-joint angles (deg) and velocities (deg/s)
        - Step counter, goals reached, self-collisions
        """
        return draw_metrics_overlay(
            self,
            frame,
            panel_x=panel_x,
            panel_y=panel_y,
            panel_w=panel_w,
        )

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the scene with an optional metrics overlay."""
        if mode == "rgb_array":
            return compose_multi_camera_frame(self)
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
