"""Crazyflie 2.1 wall-maze obstacle-avoidance reach environment.

Extends the reach-to-random-goals task with a procedurally generated maze
of axis-aligned walls.  Each episode spawns 2-3 wall barriers with gaps
the drone must navigate through.  Some gaps contain partial-height walls
that require altitude changes (fly-over or fly-under).

The rangefinder casts rays via ``mj_ray`` using MuJoCo geom-group filtering
so that the drone's own geometry is invisible to the sensor (group 0 = drone,
group 1 = walls + floor + arena walls).

Two rangefinder modes are supported:
    ``"multi_ranger"`` – 5 rays matching the real Crazyflie Multi-ranger deck
                         (front, back, left, right, up).
    ``"lidar"``        – N evenly-spaced horizontal rays + up/down verticals.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import gymnasium
import mujoco
import numpy as np

from mujoco_robot.envs.step_result import StepResult
from mujoco_robot.tasks.crazyflie.crazyflie_reach_env import (
    CrazyflieReachEnv,
    CrazyflieReachGymnasium,
    _DEFAULT_MODEL,
)


# ======================================================================
#  Wall defaults
# ======================================================================
_WALL_THICKNESS = 0.015          # 1.5 cm thick
_WALL_DEFAULT_SIZE = "0.015 0.30 0.40"

# Colours per variant  (alpha ≤ 0.40 so the drone stays visible through walls)
_RGBA_FULL      = (0.70, 0.25, 0.20, 0.85)   # brick red, mostly opaque
_RGBA_FLY_OVER  = (0.85, 0.65, 0.20, 0.75)   # golden, semi-transparent
_RGBA_FLY_UNDER = (0.50, 0.25, 0.55, 0.75)   # purple, semi-transparent


# ======================================================================
#  Gymnasium wrapper
# ======================================================================
class CrazyflieObstacleGymnasium(gymnasium.Env):
    """Gymnasium wrapper around :class:`CrazyflieObstacleEnv`."""

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        seed: int | None = None,
        render: bool = False,
        render_mode: str | None = None,
        time_limit: int = 800,
        model_path: str = _DEFAULT_MODEL,
        actuator_profile: str = "crazyflie",
        **env_kwargs,
    ):
        resolved = render_mode or ("rgb_array" if render else None)
        if resolved not in {None, "rgb_array", "human"}:
            raise ValueError("render_mode must be one of: None, 'rgb_array', 'human'")

        self.base = CrazyflieObstacleEnv(
            model_path=model_path,
            actuator_profile=actuator_profile,
            render_size=(640, 480) if resolved == "rgb_array" else (240, 180),
            time_limit=time_limit,
            seed=seed,
            **env_kwargs,
        )
        self.action_space = gymnasium.spaces.Box(
            -1.0, 1.0, shape=(self.base.action_dim,), dtype=np.float32,
        )
        self.observation_space = gymnasium.spaces.Box(
            -np.inf, np.inf,
            shape=(self.base.observation_dim,),
            dtype=np.float32,
        )
        self.render_mode = resolved
        self._human_viewer = None

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options=None):
        obs = self.base.reset(seed=seed)
        if self.render_mode == "human":
            self.render()
        return obs.astype(np.float32), {}

    def step(self, action):
        res: StepResult = self.base.step(action)
        terminated = bool(
            res.info.get("crashed", False)
            or res.info.get("battery_depleted", False)
            or res.info.get("out_of_bounds", False)
            or res.info.get("excessive_tilt", False)
            or res.info.get("obstacle_collision", False)
            or res.info.get("ceiling_collision", False)
            or res.info.get("terminated_on_goal", False)
        )
        truncated = bool(res.info.get("time_out", False) and not terminated)
        if self.render_mode == "human":
            self.render()
        return res.obs, res.reward, terminated, truncated, res.info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.base.render(mode="rgb_array")
        if self.render_mode == "human":
            if self._human_viewer is None:
                try:
                    import mujoco.viewer as mj_viewer
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError(
                        "Human rendering requires mujoco.viewer with GUI support."
                    ) from exc
                self._human_viewer = mj_viewer.launch_passive(
                    self.base.model, self.base.data,
                )
            if hasattr(self._human_viewer, "is_running") and not self._human_viewer.is_running():
                self._human_viewer.close()
                self._human_viewer = None
                return None
            if hasattr(self._human_viewer, "sync"):
                self._human_viewer.sync()
            return None
        return None

    def close(self):
        if self._human_viewer is not None:
            self._human_viewer.close()
            self._human_viewer = None
        self.base.close()


# ======================================================================
#  Core environment (subclasses CrazyflieReachEnv)
# ======================================================================
class CrazyflieObstacleEnv(CrazyflieReachEnv):
    """Motor-level Crazyflie 2.1 reach task with procedural wall-maze.

    Inherits **all** physics, motor dynamics, aerodynamics, battery
    modelling, and rendering infrastructure from
    :class:`CrazyflieReachEnv`.  This class adds:

    * **Wall-maze** -- 2-3 axis-aligned wall barriers with gaps.  Some
      gaps contain partial-height walls requiring altitude changes
      (fly-over or fly-under).  Walls are injected as mocap bodies
      whose size is set dynamically each :meth:`reset`.
    * **Multi-ray rangefinder** -- ``mj_ray``-based distance sensor with
      configurable ray count, max range, and Gaussian noise.
    * **Modified reward** -- five-component reward: progress toward goal,
      obstacle avoidance, energy efficiency, stability, and task
      completion (goal bonus / collision / timeout).
    * **Extended observations** -- parent obs concatenated with normalised
      rangefinder readings (0 = touching, 1 = max range / clear).
    """

    def __init__(
        self,
        # ---- maze parameters ----
        n_wall_slots: int = 12,
        n_barriers: Tuple[int, int] = (2, 3),
        gap_width: float = 0.22,
        partial_wall_prob: float = 0.30,
        wall_spawn_clearance: float = 0.20,
        fixed_layout: bool = True,
        ceiling_height: float = 0.75,
        terminate_on_goal: bool = False,
        # ---- rangefinder parameters ----
        rangefinder_mode: str = "lidar",
        n_lidar_rays: int = 16,
        rangefinder_max_range: float = 1.0,
        rangefinder_noise_std: float = 0.02,
        # ---- reward weights (5-component) ----
        w_progress: float = 5.0,
        w_obstacle: float = 0.5,
        w_energy: float = 0.2,
        w_stability: float = 0.3,
        w_task: float = 1.0,
        # ---- reward thresholds ----
        goal_bonus: float = 10.0,
        time_bonus_max: float = 5.0,
        collision_penalty: float = 20.0,
        timeout_penalty: float = 5.0,
        v_optimal: float = 0.5,
        v_max: float = 2.0,
        a_max: float = 15.0,
        obstacle_collision_zone: float = 0.08,
        obstacle_danger_zone: float = 0.25,
        obstacle_warning_zone: float = 0.50,
        # ---- structured goal pool ----
        goal_pool: Optional[List[Tuple[float, float, float]]] = None,
        **kwargs,
    ) -> None:
        # ---- Store BEFORE super().__init__ (which calls _load_model_xml) --
        self._n_wall_slots = int(max(0, n_wall_slots))
        self._n_barriers = (
            int(max(0, n_barriers[0])),
            int(max(n_barriers[0], n_barriers[1])),
        )
        self._gap_width = float(max(0.10, gap_width))
        self._partial_wall_prob = float(np.clip(partial_wall_prob, 0.0, 1.0))
        self._wall_spawn_clearance = float(max(0.05, wall_spawn_clearance))
        self._fixed_layout = bool(fixed_layout)
        self._ceiling_height = float(max(0, ceiling_height))
        self._terminate_on_goal = bool(terminate_on_goal)

        self._rangefinder_mode = str(rangefinder_mode)
        self._n_lidar_rays = int(max(1, n_lidar_rays))
        self._rangefinder_max_range = float(max(0.1, rangefinder_max_range))
        self._rangefinder_noise_std = float(max(0, rangefinder_noise_std))

        self._w_progress = float(w_progress)
        self._w_obstacle = float(w_obstacle)
        self._w_energy = float(w_energy)
        self._w_stability = float(w_stability)
        self._w_task = float(w_task)
        self._goal_bonus = float(max(0, goal_bonus))
        self._time_bonus_max = float(max(0, time_bonus_max))
        self._collision_penalty = float(max(0, collision_penalty))
        self._timeout_penalty = float(max(0, timeout_penalty))
        self._v_optimal = float(max(0, v_optimal))
        self._v_max = float(max(0.01, v_max))
        self._a_max = float(max(0.01, a_max))
        self._obstacle_collision_zone = float(max(0, obstacle_collision_zone))
        self._obstacle_danger_zone = float(max(obstacle_collision_zone, obstacle_danger_zone))
        self._obstacle_warning_zone = float(max(obstacle_danger_zone, obstacle_warning_zone))

        # ---- Structured goal pool ----
        # When set, goals cycle through the pool (shuffled each episode)
        # instead of being sampled randomly.
        if goal_pool is not None:
            self._goal_pool: Optional[np.ndarray] = np.array(
                goal_pool, dtype=np.float64
            )
        else:
            self._goal_pool = None
        self._goal_pool_shuffled: Optional[np.ndarray] = None
        self._goal_pool_idx: int = 0

        # ---- Build model via parent (calls overridden _load_model_xml) ----
        super().__init__(**kwargs)

        # ---- Override workspace limits for ceiling ----
        if self._ceiling_height > 0:
            self.workspace_z = (self.workspace_z[0], self._ceiling_height + 0.10)
            self.goal_z_range = (
                self.goal_z_range[0],
                min(self.goal_z_range[1], self._ceiling_height - 0.05),
            )

        # ---- Max distance for normalising progress shaping ----
        self._max_distance = float(np.sqrt(
            (2 * self.workspace_xy) ** 2 * 2
            + (self.workspace_z[1] - self.workspace_z[0]) ** 2
        ))

        # ---- Post-init: resolve wall body / geom / mocap IDs ------------
        self._wall_body_ids: list[int] = []
        self._wall_mocap_ids: list[int] = []
        self._wall_geom_ids: set[int] = set()
        self._wall_geom_ids_list: list[int] = []

        for i in range(self._n_wall_slots):
            bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"wall_{i}",
            )
            gid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"wall_{i}_geom",
            )
            if bid < 0 or gid < 0:
                continue
            self._wall_body_ids.append(bid)
            self._wall_mocap_ids.append(int(self.model.body_mocapid[bid]))
            self._wall_geom_ids.add(gid)
            self._wall_geom_ids_list.append(gid)
        self._ceiling_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ceiling",
        )

        # ---- Rangefinder setup -------------------------------------------
        self._rf_dirs_body = self._build_ray_directions()
        self._n_rf_total = len(self._rf_dirs_body)

        # Geom-group mask: only test group 1 (walls + floor + arena)
        self._rf_geomgroup = np.zeros(6, dtype=np.uint8)
        self._rf_geomgroup[1] = 1
        self._cached_rangefinder: Optional[np.ndarray] = None

        # ---- Active wall state (updated each reset) ---------------------
        self._n_active_walls = 0
        self._active_wall_positions = np.zeros(
            (self._n_wall_slots, 3), dtype=np.float64,
        )
        self._active_wall_sizes = np.zeros(
            (self._n_wall_slots, 3), dtype=np.float64,
        )

        # ---- Fixed maze layout (generated once in __init__) --------------
        self._barrier_specs: list[dict] = []
        if self._fixed_layout:
            self._fixed_maze_walls = self._generate_maze()
        else:
            self._fixed_maze_walls = None

        # ---- FPV + side + top cameras (created lazily on first render()) ---
        self._camera_names = ["cf_fpv", "cf_side", "cf_top"]
        self._renderers = []  # allocated lazily on first render() call

    # ==================================================================
    #  XML injection
    # ==================================================================
    def _load_model_xml(self, model_path: str) -> str:
        """Override: inject wall slots and assign geom groups."""
        xml_str = super()._load_model_xml(model_path)
        root = ET.fromstring(xml_str)

        worldbody = root.find("worldbody")
        if worldbody is None:
            return xml_str

        # --- Camera overrides ------------------------------------------------
        # cf_side  : FIXED 3/4-angle view of the entire arena.
        # cf_top   : fixed bird's-eye for minimap reference.
        _cam_overrides = {
            "cf_side": {
                "mode": "fixed",
                "pos": "1.6 -1.0 1.4",
                "xyaxes": "0.55 0.84 0 -0.27 0.18 0.95",
                "fovy": "70",
            },
            "cf_top": {
                "mode": "fixed",
                "pos": "0 0 2.0",
                "xyaxes": "1 0 0 0 1 0",
                "fovy": "65",
            },
        }
        for cam in worldbody.findall("camera"):
            name = cam.get("name", "")
            if name in _cam_overrides:
                for attr, val in _cam_overrides[name].items():
                    cam.set(attr, val)

        # --- FPV camera: ATTACHED to the drone body so it rotates with it.
        #     pos is relative to the cf2 body frame:
        #       Y = forward (+Y body), Z = up, X = right.
        #     Slightly behind, slightly above, looking forward & ~10° down.
        cf2_body = worldbody.find(".//body[@name='cf2']")
        if cf2_body is not None:
            ET.SubElement(
                cf2_body, "camera",
                name="cf_fpv",
                pos="0 -0.15 0.06",
                xyaxes="1 0 0 0 0.17 0.98",
                fovy="90",
            )

        # --- Move floor & arena walls to geom group 1 (rangefinder-visible)
        for geom in worldbody.findall("geom"):
            name = geom.get("name", "")
            if name in (
                "floor", "arena_north", "arena_south",
                "arena_east", "arena_west",
            ):
                geom.set("group", "1")

        # --- Inject N wall slot mocap bodies (initially underground) ------
        rgba_str = f"{_RGBA_FULL[0]} {_RGBA_FULL[1]} {_RGBA_FULL[2]} {_RGBA_FULL[3]}"
        for i in range(self._n_wall_slots):
            wall_body = ET.SubElement(
                worldbody, "body",
                name=f"wall_{i}", mocap="true", pos="0 0 -5",
            )
            ET.SubElement(
                wall_body, "geom",
                name=f"wall_{i}_geom",
                type="box",
                size=_WALL_DEFAULT_SIZE,
                rgba=rgba_str,
                contype="1", conaffinity="1",
                group="1",
                margin="0.015",
                friction="0.9 0.05 0.02",
                solref="0.005 1",
                solimp="0.9 0.95 0.001",
            )

        # --- Inject ceiling (transparent, collision-enabled) --------------
        if self._ceiling_height > 0:
            ET.SubElement(
                worldbody, "geom",
                name="ceiling",
                type="box",
                pos=f"0 0 {self._ceiling_height}",
                size="1.1 1.1 0.005",
                rgba="0.7 0.85 1.0 0.12",
                contype="1", conaffinity="1",
                group="1",
            )

        return ET.tostring(root, encoding="unicode")

    # ==================================================================
    #  Rangefinder
    # ==================================================================
    def _build_ray_directions(self) -> np.ndarray:
        """Unit-length ray directions in **body frame**."""
        dirs: list[list[float]] = []

        if self._rangefinder_mode == "multi_ranger":
            dirs = [
                [1, 0, 0],     # front  (+X body)
                [-1, 0, 0],    # back
                [0, 1, 0],     # left   (+Y body)
                [0, -1, 0],    # right
                [0, 0, 1],     # up
            ]
        else:
            # "lidar" -- horizontal ring + vertical pair
            for i in range(self._n_lidar_rays):
                angle = 2.0 * math.pi * i / self._n_lidar_rays
                dirs.append([math.cos(angle), math.sin(angle), 0.0])
            dirs.append([0.0, 0.0, 1.0])    # up
            dirs.append([0.0, 0.0, -1.0])   # down

        arr = np.array(dirs, dtype=np.float64)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-8)

    @staticmethod
    def _yaw_only_matrix(rot: np.ndarray) -> np.ndarray:
        """Extract a yaw-only rotation from a full 3×3 rotation matrix.

        Projects the body-X axis onto the world XY plane and builds a
        rotation about Z.  This keeps the horizontal lidar ring level
        regardless of pitch/roll — matching real mechanical lidars.
        """
        fwd = rot[:, 0].copy()          # body-X in world frame
        fwd[2] = 0.0                    # project onto XY plane
        norm = np.linalg.norm(fwd)
        if norm < 1e-8:                 # pathological – drone is vertical
            return np.eye(3)
        fwd /= norm
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(up, fwd)
        right /= np.linalg.norm(right) + 1e-12
        return np.column_stack([fwd, right, up])  # columns = new axes

    def _compute_rangefinder(self) -> np.ndarray:
        """Cast rays from the drone body; return normalised distances [0, 1].

        Horizontal rays rotate with yaw only (level plane, like a real
        lidar).  Vertical rays (up/down) are always world-frame.

        0 = surface contact, 1 = max range (or no hit).
        """
        pos = self.data.xpos[self.cf_body_id].copy()
        rot = self.data.xmat[self.cf_body_id].reshape(3, 3)
        yaw_rot = self._yaw_only_matrix(rot)
        geomid_out = np.array([-1], dtype=np.int32)

        readings = np.ones(self._n_rf_total, dtype=np.float32)

        for i, dir_body in enumerate(self._rf_dirs_body):
            is_vertical = abs(dir_body[2]) > 0.9
            if is_vertical:
                # Up / down — always world-frame (unaffected by tilt)
                dir_world = dir_body.copy()
            else:
                # Horizontal ring — yaw only, stays level
                dir_world = (yaw_rot @ dir_body).astype(np.float64)
            dist = mujoco.mj_ray(
                self.model, self.data,
                pos, dir_world,
                self._rf_geomgroup, 1, -1, geomid_out,
            )
            if dist >= 0:
                readings[i] = float(
                    np.clip(dist / self._rangefinder_max_range, 0.0, 1.0)
                )

        # Gaussian sensor noise (realistic for VL53L1x)
        if self._rangefinder_noise_std > 0.0:
            noise = self._rng.normal(
                0.0, self._rangefinder_noise_std / self._rangefinder_max_range,
                size=self._n_rf_total,
            )
            readings = np.clip(
                readings + noise.astype(np.float32), 0.0, 1.0,
            )

        return readings

    # ==================================================================
    #  Collision detection
    # ==================================================================
    def _check_obstacle_collision(self) -> bool:
        """Return True if any drone geom is in contact with a wall geom,
        OR if any horizontal rangefinder ray detects an obstacle closer
        than 2 cm (proximity fallback for tunnelling prevention)."""
        # 1) MuJoCo contact-pair check (c.dist <= 0 means actual penetration;
        #    positive dist means geoms are within geom margin but not touching)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.dist > 0:
                continue  # within margin zone but not actually touching
            g1, g2 = int(c.geom1), int(c.geom2)
            if (
                (g1 in self._drone_geom_ids and g2 in self._wall_geom_ids)
                or (g2 in self._drone_geom_ids and g1 in self._wall_geom_ids)
            ):
                return True
        # 2) Proximity fallback — horizontal rays only (exclude vertical rays)
        if hasattr(self, "_cached_rangefinder") and self._cached_rangefinder is not None:
            rf = self._cached_rangefinder
            # multi_ranger: [front, back, left, right, up] — 1 vertical at end
            # lidar:        [h0..hN-1, up, down]            — 2 verticals at end
            if self._rangefinder_mode == "multi_ranger":
                n_horiz = len(rf) - 1
            else:
                n_horiz = len(rf) - 2
            if n_horiz > 0:
                proximity_threshold = 0.02 / self._rangefinder_max_range  # 2 cm
                if float(np.min(rf[:n_horiz])) < proximity_threshold:
                    return True
        return False

    def _check_ceiling_collision(self) -> bool:
        """Return True if any drone geom is in contact with the ceiling geom."""
        if self._ceiling_geom_id < 0:
            return False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (
                (g1 in self._drone_geom_ids and g2 == self._ceiling_geom_id)
                or (g2 in self._drone_geom_ids and g1 == self._ceiling_geom_id)
            ):
                return True
        return False

    # ==================================================================
    #  Maze generation
    # ==================================================================
    def _generate_maze(self) -> list[dict]:
        """Procedurally generate a wall-maze layout.

        Creates 2-3 wall barriers that span the arena, each with 1-2
        gaps.  Barriers alternate between Y-aligned (blocking X movement)
        and X-aligned (blocking Y movement) to form a maze-like structure.

        Some gaps may contain partial-height walls:

        * **fly_over** -- short wall from the floor; drone passes above.
        * **fly_under** -- elevated wall; drone passes beneath.

        Returns a list of wall specs (dicts with ``pos``, ``size``,
        ``variant``), capped at ``n_wall_slots``.
        """
        # Use a deterministic seed so the maze layout is identical across
        # all env instances (different seeds) when fixed_layout is True.
        _swap_rng = self._fixed_layout
        if _swap_rng:
            _original_rng = self._rng
            self._rng = np.random.default_rng(12345)

        walls: list[dict] = []
        self._barrier_specs = []
        n_barriers = int(
            self._rng.integers(self._n_barriers[0], self._n_barriers[1] + 1)
        )

        for b_idx in range(n_barriers):
            # Alternate orientation for a proper maze feel
            is_y_aligned = (b_idx % 2 == 0)

            # Spread barriers evenly across the arena with small jitter
            frac = (b_idx + 1.0) / (n_barriers + 1.0)
            perp_range = 0.38
            perp_base = -perp_range + frac * 2.0 * perp_range
            jitter = float(self._rng.uniform(-0.06, 0.06))
            perp_pos = perp_base + jitter

            # Enforce minimum distance from spawn (0, 0)
            if abs(perp_pos) < self._wall_spawn_clearance:
                perp_pos = (
                    self._wall_spawn_clearance * (1.0 if perp_pos >= 0 else -1.0)
                )

            # Track barrier for goal-behind-walls sampling
            self._barrier_specs.append({
                "is_y_aligned": is_y_aligned,
                "perp_pos": perp_pos,
            })

            # Wall spans across the arena
            span_lo, span_hi = -0.50, 0.50

            # Create 1-2 gaps per barrier
            n_gaps = int(self._rng.integers(1, 3))
            gap_centers: list[float] = []
            for g in range(n_gaps):
                g_frac = (g + 0.5) / n_gaps
                g_base = span_lo + g_frac * (span_hi - span_lo)
                g_jitter = float(self._rng.uniform(-0.10, 0.10))
                gap_centers.append(g_base + g_jitter)
            gap_centers.sort()

            # Merge overlapping gaps
            if (
                n_gaps >= 2
                and gap_centers[1] - gap_centers[0] < self._gap_width + 0.08
            ):
                gap_centers = [gap_centers[0]]

            # Build wall segments around gaps
            current = span_lo
            for gc in gap_centers:
                gap_lo = gc - self._gap_width / 2
                gap_hi = gc + self._gap_width / 2

                # Solid segment before this gap
                if gap_lo - current > 0.06:
                    walls.append(
                        self._make_wall_segment(
                            is_y_aligned, perp_pos, current, gap_lo, "full",
                        )
                    )

                # Optionally place a partial wall inside the gap
                if self._rng.random() < self._partial_wall_prob:
                    variant = self._rng.choice(["fly_over", "fly_under"])
                    inner_lo = gc - self._gap_width / 2 + 0.02
                    inner_hi = gc + self._gap_width / 2 - 0.02
                    walls.append(
                        self._make_wall_segment(
                            is_y_aligned, perp_pos, inner_lo, inner_hi, variant,
                        )
                    )

                current = gap_hi

            # Solid segment after the last gap
            if span_hi - current > 0.06:
                walls.append(
                    self._make_wall_segment(
                        is_y_aligned, perp_pos, current, span_hi, "full",
                    )
                )

        # Filter out any wall whose bounding box overlaps the spawn zone.
        # The drone spawns at (0, 0, ~0.35) and has a body radius ≈0.05 m.
        drone_radius = 0.07  # conservative envelope
        spawn_xy = np.array([0.0, 0.0])
        safe_walls: list[dict] = []
        for w in walls:
            p, s = w["pos"], w["size"]
            # Wall box: [p[0]-s[0], p[0]+s[0]] x [p[1]-s[1], p[1]+s[1]]
            # Check if the drone circle overlaps this box in XY
            # Clamp spawn_xy to wall box, compute distance
            closest_x = np.clip(spawn_xy[0], p[0] - s[0], p[0] + s[0])
            closest_y = np.clip(spawn_xy[1], p[1] - s[1], p[1] + s[1])
            dist = np.sqrt((spawn_xy[0] - closest_x) ** 2
                           + (spawn_xy[1] - closest_y) ** 2)
            if dist >= drone_radius:
                safe_walls.append(w)

        if _swap_rng:
            self._rng = _original_rng

        return safe_walls[: self._n_wall_slots]

    def _make_wall_segment(
        self,
        is_y_aligned: bool,
        perp_pos: float,
        along_lo: float,
        along_hi: float,
        variant: str,
    ) -> dict:
        """Create one wall segment specification.

        Parameters
        ----------
        is_y_aligned : bool
            True = wall runs along Y (blocks X movement).
        perp_pos : float
            Position on the perpendicular axis.
        along_lo, along_hi : float
            Extent of the segment along the wall axis.
        variant : str
            ``"full"`` | ``"fly_over"`` | ``"fly_under"``.
        """
        along_center = (along_lo + along_hi) / 2
        half_span = max(0.03, (along_hi - along_lo) / 2)

        if variant == "fly_over":
            # Short wall from floor -- drone flies above
            half_height = float(self._rng.uniform(0.08, 0.15))
            z_center = half_height
        elif variant == "fly_under":
            # Elevated wall -- drone flies beneath
            half_height = float(self._rng.uniform(0.12, 0.20))
            z_center = float(self._rng.uniform(0.48, 0.58))
        else:
            # Full-height wall — extends to ceiling when present
            if self._ceiling_height > 0:
                half_height = self._ceiling_height / 2.0
            else:
                half_height = float(self._rng.uniform(0.35, 0.45))
            z_center = half_height

        if is_y_aligned:
            pos = [perp_pos, along_center, z_center]
            size = [_WALL_THICKNESS, half_span, half_height]
        else:
            pos = [along_center, perp_pos, z_center]
            size = [half_span, _WALL_THICKNESS, half_height]

        return {
            "pos": np.array(pos, dtype=np.float64),
            "size": np.array(size, dtype=np.float64),
            "variant": variant,
        }

    # ==================================================================
    #  Obstacle placement
    # ==================================================================
    def _reset_obstacles(self) -> None:
        """Apply the maze layout to the mocap wall slots.

        When ``fixed_layout`` is True the same wall configuration is
        reused every episode; otherwise a fresh maze is generated.
        """
        if self._fixed_layout and self._fixed_maze_walls is not None:
            walls = self._fixed_maze_walls
        else:
            walls = self._generate_maze()
        self._n_active_walls = len(walls)

        for i, wall in enumerate(walls):
            mid = self._wall_mocap_ids[i]
            gid = self._wall_geom_ids_list[i]
            self.data.mocap_pos[mid] = wall["pos"]
            self.model.geom_size[gid] = wall["size"]
            self._active_wall_positions[i] = wall["pos"]
            self._active_wall_sizes[i] = wall["size"]

            # Colour by variant
            if wall["variant"] == "fly_over":
                self.model.geom_rgba[gid] = _RGBA_FLY_OVER
            elif wall["variant"] == "fly_under":
                self.model.geom_rgba[gid] = _RGBA_FLY_UNDER
            else:
                self.model.geom_rgba[gid] = _RGBA_FULL

        # Deactivate remaining slots (underground)
        for i in range(self._n_active_walls, len(self._wall_mocap_ids)):
            mid = self._wall_mocap_ids[i]
            self.data.mocap_pos[mid] = [0.0, 0.0, -5.0]
            self._active_wall_positions[i] = [0.0, 0.0, -5.0]
            self._active_wall_sizes[i] = [0.0, 0.0, 0.0]

    # ==================================================================
    #  Goal sampling (always behind a wall barrier)
    # ==================================================================
    def _sample_goal(self) -> np.ndarray:
        """Override: sample goal on the far side of a wall barrier.

        Ensures the drone must navigate through at least one gap to
        reach the target.  Falls back to parent sampling when no
        barriers exist.
        """
        if not self._barrier_specs:
            return super()._sample_goal()

        # Pick a random barrier the goal must be placed behind
        idx = int(self._rng.integers(len(self._barrier_specs)))
        b = self._barrier_specs[idx]
        pp = b["perp_pos"]
        margin = 0.08  # minimum gap beyond barrier surface

        xy_range = self.goal_xy_range

        if b["is_y_aligned"]:
            # Barrier runs along Y, blocks X movement; perpendicular is X
            if pp >= 0:
                x_lo, x_hi = pp + margin, xy_range
            else:
                x_lo, x_hi = -xy_range, pp - margin
            if x_lo >= x_hi:
                return super()._sample_goal()
            x = float(self._rng.uniform(x_lo, x_hi))
            y = float(self._rng.uniform(-xy_range, xy_range))
        else:
            # Barrier runs along X, blocks Y movement; perpendicular is Y
            if pp >= 0:
                y_lo, y_hi = pp + margin, xy_range
            else:
                y_lo, y_hi = -xy_range, pp - margin
            if y_lo >= y_hi:
                return super()._sample_goal()
            x = float(self._rng.uniform(-xy_range, xy_range))
            y = float(self._rng.uniform(y_lo, y_hi))

        z = float(self._rng.uniform(self.goal_z_range[0], self.goal_z_range[1]))
        return np.array([x, y, z], dtype=float)

    # ==================================================================
    #  Overridden API
    # ==================================================================
    @property
    def observation_dim(self) -> int:
        # Parent obs + rangefinder(N)
        return super().observation_dim + self._n_rf_total

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        # Parent reset: spawns drone, samples goal, clears odom, etc.
        super().reset(seed=seed)
        # Place walls (needs goal_pos from parent reset)
        self._reset_obstacles()
        # Goal assignment priority:
        #   1. Structured pool (shuffled each episode) — overrides everything.
        #   2. Random barrier-aware sampling for random layouts.
        #   3. Goal already set by parent reset (fixed_layout=True).
        if self._goal_pool is not None:
            shuffled = self._goal_pool.copy()
            self._rng.shuffle(shuffled)
            self._goal_pool_shuffled = shuffled
            self._goal_pool_idx = 0
            self._goal_pos = shuffled[0].copy()
            self._update_goal_marker()
        elif not self._fixed_layout:
            self._goal_pos = self._sample_goal()
            self._update_goal_marker()
        mujoco.mj_forward(self.model, self.data)
        # Potential-based shaping: initialise previous distance
        pos = self.data.xpos[self.cf_body_id]
        self._prev_distance = float(np.linalg.norm(self._goal_pos - pos))
        # Velocity / acceleration history for energy & stability terms
        self._prev_lin_vel = np.zeros(3, dtype=np.float64)
        self._prev_accel = np.zeros(3, dtype=np.float64)
        self._cached_rangefinder = None
        return self._observe()

    def _observe(self) -> np.ndarray:
        """Parent obs + rangefinder."""
        base_obs = super()._observe()
        if self._cached_rangefinder is None:
            rf = self._compute_rangefinder()
        else:
            rf = self._cached_rangefinder
            self._cached_rangefinder = None
        return np.concatenate([base_obs, rf]).astype(np.float32)

    # ==================================================================
    #  Reward
    # ==================================================================
    def _compute_reward_and_info(self):
        """Five-component reward for wall-maze navigation.

        1. **R_progress** — distance progress + directional alignment
        2. **R_obstacle** — zone-based proximity penalty (rangefinder)
        3. **R_energy**   — thrust / balance / velocity / acceleration
        4. **R_stability** — angular velocity / tilt / jerk
        5. **R_task**     — goal bonus + time bonus + collision + timeout

        All components are scaled to roughly [-1, +1] range so that the
        component weights (w_*) directly control the relative importance.

        Total = w_progress·R_progress + w_obstacle·R_obstacle
              + w_energy·R_energy + w_stability·R_stability
              + w_task·R_task
        """
        # ---- State ----
        pos, quat, lin_vel_world, ang_vel_world, rot_mat = self._body_state()
        pos_error = float(np.linalg.norm(self._goal_pos - pos))
        body_z = rot_mat[:, 2]
        tilt_cos = float(body_z[2])
        tilt_rad = float(
            math.acos(np.clip(tilt_cos, -1.0, 1.0))
        )
        tilt_deg = math.degrees(tilt_rad)
        lin_speed = float(np.linalg.norm(lin_vel_world))
        ang_speed = float(np.linalg.norm(ang_vel_world))

        # ---- Goal reach check (replicated from parent) ----
        near_goal = pos_error < self.reach_threshold
        if near_goal:
            self._reach_hold_counter += 1
        else:
            self._reach_hold_counter = 0

        goal_just_reached = False
        if self._reach_hold_counter >= self.reach_hold_steps:
            goal_just_reached = True
            self._goals_reached += 1
            self._reach_hold_counter = 0
            if not self._terminate_on_goal:
                if self._goal_pool_shuffled is not None:
                    self._goal_pool_idx = (
                        (self._goal_pool_idx + 1) % len(self._goal_pool_shuffled)
                    )
                    self._goal_pos = self._goal_pool_shuffled[
                        self._goal_pool_idx
                    ].copy()
                else:
                    self._goal_pos = self._sample_goal()
                self._update_goal_marker()
                self._prev_distance = float(
                    np.linalg.norm(self._goal_pos - pos)
                )

        # ---- Rangefinder (computed once, used by collision + reward) ----
        rf = self._compute_rangefinder()
        self._cached_rangefinder = rf.copy()

        # ---- Termination checks (replicated from parent) ----
        crashed = bool(self._is_crashed_with_floor() or pos[2] <= self.workspace_z[0])
        out_of_bounds = bool(
            abs(pos[0]) > self.workspace_xy
            or abs(pos[1]) > self.workspace_xy
            or pos[2] > self.workspace_z[1]
        )
        excessive_tilt = bool(tilt_deg > self.termination_tilt_deg)
        battery_depleted = bool(self.battery_soc <= 0.05)
        time_out = bool(self.step_id >= self.time_limit)
        obstacle_collision = self._check_obstacle_collision()
        ceiling_collision = self._check_ceiling_collision()
        collision_detected = crashed or obstacle_collision or ceiling_collision
        terminated_on_goal = bool(self._terminate_on_goal and goal_just_reached)

        dt = max(self.dt_control, 1e-6)

        # ==================== 1. R_progress  (range ~ [-1, +1]) ===========
        # Distance progress — potential-based, using raw metres
        d_prev = self._prev_distance
        d_curr = pos_error
        # Scale by typical goal distance (~0.5m) so a 0.005 m/step
        # approach gives R_distance ~ +0.01 per step
        R_distance = (d_prev - d_curr) * 2.0  # ≈ 1/typical_dist
        self._prev_distance = d_curr

        # Directional alignment (already in [0, 1])
        goal_vec = self._goal_pos - pos
        dist = float(np.linalg.norm(goal_vec))
        if dist > 0.01:
            goal_dir = goal_vec / dist
            vel_toward_goal = float(np.dot(lin_vel_world, goal_dir))
        else:
            vel_toward_goal = 0.0
        if dist > 0.01:
            # tanh gives a smooth signal in [-1, 1] regardless of speed,
            # rewarding any drift toward goal and penalising moving away.
            R_direction = float(np.tanh(vel_toward_goal / 0.3))
        else:
            R_direction = 0.0

        # Exponential approach bonus: +0.8 at goal, ~0.11 at 0.5 m.
        # NOT gated by obstacle clearance — obstacle avoidance is handled by
        # R_obstacle separately; gating here suppresses the goal gradient
        # whenever any wall is nearby, which kills progressive learning.
        R_approach = 0.8 * math.exp(-pos_error / 0.25)

        # Near-goal holding bonus (mirrors crazyflie_reach_env)
        R_near_goal = 0.08 if near_goal else 0.0

        # Alive bonus: small constant survival signal so stable hover is
        # always positive, even in the presence of energy/stability penalties.
        R_alive = 0.05

        R_progress = (
            0.4 * R_distance
            + 0.2 * R_direction
            + 0.2 * R_approach
            + 0.1 * R_near_goal
            + 0.1 * R_alive
        )

        # ==================== 2. R_obstacle  (range ~ [-1, 0]) ============
        # rf is already computed above (line 887) and cached; reuse it.
        min_rf_norm = float(np.min(rf))
        min_range_m = min_rf_norm * self._rangefinder_max_range

        # Zone thresholds are interpreted as *normalised* rangefinder
        # fractions [0, 1] rather than absolute metres, matching the
        # actual sensor range.  This prevents the pathological case
        # where all rays fall inside the warning zone.
        cz = self._obstacle_collision_zone   # e.g. 0.10 (10% of max range)
        dz = self._obstacle_danger_zone      # e.g. 0.30
        wz = self._obstacle_warning_zone     # e.g. 0.60

        # Vectorised zone penalties — avoids a Python loop over all rays.
        in_cz = rf < cz
        in_dz = (~in_cz) & (rf < dz)
        in_wz = (~in_cz) & (~in_dz) & (rf < wz)
        penalties = np.zeros(len(rf), dtype=np.float32)
        penalties[in_cz] = -1.0
        if in_dz.any():
            penalties[in_dz] = -0.5 * np.exp(
                -3.0 * (rf[in_dz] - cz) / max(dz - cz, 1e-6)
            )
        if in_wz.any():
            penalties[in_wz] = -0.1 * (wz - rf[in_wz]) / max(wz - dz, 1e-6)
        mean_penalty = float(np.mean(penalties))
        worst_penalty = float(penalties.min())
        # Blend worst ray (60%) with mean penalty (40%) so the agent gets a
        # graded signal: a single wall ray doesn't wipe out the progress signal
        # while the closest obstacle still dominates.
        R_obstacle_val = 0.6 * worst_penalty + 0.4 * mean_penalty

        # ==================== 3. R_energy  (range ~ [-1, 0]) ==============
        rotor_norm = self._last_motor_cmd_norm

        # Thrust penalty (quadratic — power ∝ thrust²)
        R_thrust = -float(np.sum(rotor_norm ** 2)) / 4.0  # in [-1, 0]

        # Thrust imbalance penalty
        mean_thrust = float(np.mean(rotor_norm))
        variance = float(np.sum((rotor_norm - mean_thrust) ** 2)) / 4.0
        R_balance = -variance  # in [-0.25, 0] typically

        # Excess velocity penalty (soft, in [-1, 0])
        if lin_speed > self._v_optimal:
            R_velocity = -min(1.0, ((lin_speed - self._v_optimal) / self._v_max) ** 2)
        else:
            R_velocity = 0.0

        # Acceleration penalty (clamped)
        accel_vec = (lin_vel_world - self._prev_lin_vel) / dt
        accel_mag = float(np.linalg.norm(accel_vec))
        R_accel = -min(1.0, (accel_mag / self._a_max) ** 2)

        R_energy = 0.4 * R_thrust + 0.2 * R_balance + 0.2 * R_velocity + 0.2 * R_accel

        # ==================== 4. R_stability  (range ~ [-1, 0]) ===========
        # Angular velocity: clamped to [-1, 0]
        R_angular = -min(1.0, (ang_speed / 5.0) ** 2)

        # Tilt: cos-based, 0 at level, -1 at 90° tilt
        R_tilt = -(1.0 - max(tilt_cos, 0.0))  # in [-1, 0]

        # Jerk (smoothness) — clamped so it can't dominate
        jerk_vec = (accel_vec - self._prev_accel) / dt
        jerk_mag = float(np.linalg.norm(jerk_vec))
        jerk_norm = min(1.0, jerk_mag / 500.0)  # 500 m/s³ as reference
        R_jerk = -jerk_norm

        R_stability = 0.3 * R_angular + 0.4 * R_tilt + 0.3 * R_jerk

        # ---- Update velocity / acceleration history ----
        self._prev_lin_vel = lin_vel_world.copy()
        self._prev_accel = accel_vec.copy()

        # ==================== 5. R_task  (sparse events) ==================
        R_success = 0.0
        R_time_bonus = 0.0
        if goal_just_reached:
            R_success = self._goal_bonus
            R_time_bonus = max(
                0.0,
                self._time_bonus_max * (1.0 - self.step_id / self.time_limit),
            )

        R_collision = -self._collision_penalty if collision_detected else 0.0
        R_timeout = -self._timeout_penalty if time_out else 0.0

        R_task = R_success + R_time_bonus + R_collision + R_timeout

        # ==================== Total ====================
        reward = (
            self._w_progress * R_progress
            + self._w_obstacle * R_obstacle_val
            + self._w_energy * R_energy
            + self._w_stability * R_stability
            + self._w_task * R_task
        )

        # ---- Done ----
        done = bool(
            crashed or out_of_bounds or excessive_tilt
            or battery_depleted or time_out
            or obstacle_collision or ceiling_collision or terminated_on_goal
        )

        info: Dict[str, float | bool | int] = {
            # Parent-compatible keys
            "time_out": time_out,
            "crashed": crashed,
            "out_of_bounds": out_of_bounds,
            "excessive_tilt": excessive_tilt,
            "battery_soc": float(self.battery_soc),
            "battery_depleted": battery_depleted,
            "pos_error": pos_error,
            "tilt_deg": float(tilt_deg),
            "lin_speed": lin_speed,
            "ang_speed": ang_speed,
            "motor_omega_mean": float(np.mean(self.motor_omega)),
            "ground_effect_mean": float(self._last_ground_effect_mean),
            "disturbance_norm": float(np.linalg.norm(self._disturbance_force_world)),
            "goals_reached": int(self._goals_reached),
            "reach_hold_steps": int(self._reach_hold_counter),
            "reach_hold_required": int(self.reach_hold_steps),
            "goal_just_reached": goal_just_reached,
            "terminate_on_goal": self._terminate_on_goal,
            "terminated_on_goal": terminated_on_goal,
            # Obstacle-specific keys
            "obstacle_collision": obstacle_collision,
            "ceiling_collision": ceiling_collision,
            "min_obstacle_range_m": min_range_m,
            "rangefinder_min_norm": min_rf_norm,
            "n_active_walls": self._n_active_walls,
            "n_active_obstacles": self._n_active_walls,  # legacy alias
            "goal_pool_idx": int(self._goal_pool_idx),
            "goal_pool_size": int(len(self._goal_pool_shuffled)) if self._goal_pool_shuffled is not None else 0,
            # Reward components
            "R_progress": float(R_progress),
            "R_obstacle": float(R_obstacle_val),
            "R_energy": float(R_energy),
            "R_stability": float(R_stability),
            "R_task": float(R_task),
            "progress": float(R_distance),
            "R_direction": float(R_direction),
            "R_approach": float(R_approach),
            "R_near_goal": float(R_near_goal),
            "speed_match": float(R_direction),  # legacy alias for eval HUD
            "stability_factor": float(np.clip(1.0 + R_stability, 0.0, 1.0)),
            "vel_toward_goal": float(vel_toward_goal),
            "obstacle_proximity_penalty": float(R_obstacle_val),
        }
        return float(reward), done, info

    # ==================================================================
    #  Render override – inject lidar rays into MuJoCo scene
    # ==================================================================
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Override: draw lidar rays as thin lines in the 3-D views."""
        if mode == "human":
            return None
        if mode != "rgb_array":
            raise ValueError("mode must be 'human' or 'rgb_array'")

        rw, rh = self.render_size
        if not self._renderers:
            self._renderers = [
                mujoco.Renderer(self.model, height=rh, width=rw)
                for _ in self._camera_names
            ]

        # Pre-compute lidar hit points for visualisation
        drone_pos = self.data.xpos[self.cf_body_id].copy()
        rot = self.data.xmat[self.cf_body_id].reshape(3, 3)
        yaw_rot = self._yaw_only_matrix(rot)
        geomid_out = np.array([-1], dtype=np.int32)

        ray_endpoints: list[tuple[np.ndarray, np.ndarray, float]] = []
        for dir_body in self._rf_dirs_body:
            is_vertical = abs(dir_body[2]) > 0.9
            if is_vertical:
                dir_world = dir_body.copy()
            else:
                dir_world = (yaw_rot @ dir_body).astype(np.float64)
            dist = mujoco.mj_ray(
                self.model, self.data,
                drone_pos, dir_world,
                self._rf_geomgroup, 1, -1, geomid_out,
            )
            if dist >= 0:
                clamped = min(dist, self._rangefinder_max_range)
                end = drone_pos + dir_world * clamped
                norm = clamped / self._rangefinder_max_range
            else:
                end = drone_pos + dir_world * self._rangefinder_max_range
                norm = 1.0
            ray_endpoints.append((drone_pos.copy(), end.copy(), norm))

        # Render each camera view with lidar rays injected into the scene
        frames = []
        for renderer, cam_name in zip(self._renderers, self._camera_names):
            renderer.update_scene(self.data, camera=cam_name)
            scn = renderer.scene
            for start, end, norm in ray_endpoints:
                if scn.ngeom >= scn.maxgeom:
                    break
                g = scn.geoms[scn.ngeom]
                # Colour: green(far) → red(close)
                rgba = np.array([
                    1.0 - norm,     # R: high when close
                    norm * 0.8,     # G: high when far
                    0.1,            # B
                    0.6,            # alpha
                ], dtype=np.float32)
                mujoco.mjv_initGeom(
                    g,
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.eye(3).flatten(),
                    rgba=rgba,
                )
                mujoco.mjv_connector(
                    g,
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    width=2.0,  # pixels
                    from_=start.astype(np.float64),
                    to=end.astype(np.float64),
                )
                scn.ngeom += 1
            frames.append(renderer.render().copy())

        fpv_frame = frames[0]    # body-attached FPV (rotates with drone)
        side_frame = frames[1]   # fixed 3/4-angle arena overview
        # frames[2] = cf_top (used by minimap, not shown directly)

        # Synthetic panels
        minimap = self._draw_minimap((rh, rw))
        instruments = self._draw_metrics_overlay((rh, rw))

        # Uniform panel title bars
        self._add_panel_label(fpv_frame, "DRONE  FPV")
        self._add_panel_label(side_frame, "3D  SIDE  VIEW")
        self._add_panel_label(minimap, "XY  TRAJECTORY")
        self._add_panel_label(instruments, "TELEMETRY")

        # Compose 2×2:  [fpv | side]  /  [minimap | instruments]
        top_row = np.concatenate([fpv_frame, side_frame], axis=1)
        bottom_row = np.concatenate([minimap, instruments], axis=1)
        composite = np.concatenate([top_row, bottom_row], axis=0)

        # Teal accent borders
        bdr = (42, 160, 190)
        composite[rh - 1:rh + 1, :] = bdr
        composite[:, rw - 1:rw + 1] = bdr
        composite[0:2, :] = bdr
        composite[-2:, :] = bdr
        composite[:, 0:2] = bdr
        composite[:, -2:] = bdr

        return composite

    # ------------------------------------------------------------------
    def _add_panel_label(self, panel: np.ndarray, text: str) -> None:
        """Burn a small title label into the top-left of a panel image."""
        self._burn_text_lines(panel, [text], x=6, y=4, scale=1,
                              color=(200, 210, 220), line_height=18)

    # ==================================================================
    #  Minimap override (show walls as rectangles)
    # ==================================================================
    def _draw_minimap(self, shape: tuple) -> np.ndarray:
        h, w = shape
        panel = np.full((h, w, 3), 25, dtype=np.uint8)

        # Re-derive the coordinate mapping (same constants as parent)
        label_h = 24
        info_h = 38
        pad = 28
        alt_bar_w = 16
        gap = 6
        map_x0 = pad
        map_y0 = label_h + 6
        map_w = w - 2 * pad - alt_bar_w - gap
        map_h = h - map_y0 - info_h
        if map_w < 40 or map_h < 40:
            return panel
        ws = self.workspace_xy * 1.15

        def _w2p(wx: float, wy: float):
            px = map_x0 + int((wx + ws) / (2.0 * ws) * map_w)
            py = map_y0 + int((ws - wy) / (2.0 * ws) * map_h)
            return (
                int(np.clip(px, 0, w - 1)),
                int(np.clip(py, 0, h - 1)),
            )

        # Draw active walls as filled rectangles
        for i in range(self._n_active_walls):
            wpos = self._active_wall_positions[i]
            wsize = self._active_wall_sizes[i]
            if wpos[2] < -1.0:
                continue

            # World-space corners (top-down)
            x_lo = wpos[0] - wsize[0]
            x_hi = wpos[0] + wsize[0]
            y_lo = wpos[1] - wsize[1]
            y_hi = wpos[1] + wsize[1]

            # Pixel corners
            px_lo, py_hi_px = _w2p(float(x_lo), float(y_lo))
            px_hi, py_lo_px = _w2p(float(x_hi), float(y_hi))

            # Ensure at least 2px wide for thin walls
            if abs(px_hi - px_lo) < 2:
                px_lo = max(0, min(px_lo, px_hi) - 1)
                px_hi = min(w - 1, max(px_lo, px_hi) + 1)
            if abs(py_hi_px - py_lo_px) < 2:
                py_lo_px = max(0, min(py_lo_px, py_hi_px) - 1)
                py_hi_px = min(h - 1, max(py_lo_px, py_hi_px) + 1)

            # Choose colour based on wall height (approx variant)
            z_top = wpos[2] + wsize[2]
            if z_top < 0.25:
                wc = (220, 170, 50)  # golden -- fly-over
            elif wpos[2] > 0.30:
                wc = (130, 70, 150)  # purple -- fly-under
            else:
                wc = (200, 60, 60)   # red -- full wall

            for py in range(min(py_lo_px, py_hi_px), max(py_lo_px, py_hi_px) + 1):
                for px in range(min(px_lo, px_hi), max(px_lo, px_hi) + 1):
                    if 0 <= py < h and 0 <= px < w:
                        panel[py, px] = wc

        return panel
