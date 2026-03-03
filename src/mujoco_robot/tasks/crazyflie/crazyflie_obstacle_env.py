"""Crazyflie 2.1 obstacle-avoidance reach environment.

Extends the reach-to-random-goals task with:
- Random obstacles injected as mocap bodies with collision geoms
- Multi-ray rangefinder sensor (inspired by Crazyflie Multi-ranger deck /
  VL53L1x Time-of-Flight sensors)
- Obstacle proximity penalty and collision termination

The rangefinder casts rays via ``mj_ray`` using MuJoCo geom-group filtering
so that the drone's own geometry is invisible to the sensor (group 0 = drone,
group 1 = obstacles + floor + arena walls).

Two rangefinder modes are supported:
    ``"multi_ranger"`` – 5 rays matching the real Crazyflie Multi-ranger deck
                         (front, back, left, right, up).
    ``"lidar"``        – N evenly-spaced horizontal rays + up/down verticals.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Tuple

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
#  Obstacle shape palette — cycled across the N slots
# ======================================================================
_OBSTACLE_DEFS: list[tuple[str, str, str]] = [
    # (geom_type, size, rgba)
    # --- Floor-to-ceiling pillars (half-height 0.35–0.50 → tops at 0.70–1.00 m) ---
    ("box",      "0.04 0.04 0.40",   "0.78 0.18 0.16 0.85"),   # tall square pillar
    ("cylinder", "0.035 0.45",        "0.72 0.24 0.14 0.85"),   # tall round pillar
    ("box",      "0.03 0.03 0.50",    "0.74 0.20 0.14 0.85"),   # very tall thin pillar
    ("cylinder", "0.03 0.38",         "0.76 0.22 0.16 0.85"),   # medium tall round
    ("cylinder", "0.025 0.42",        "0.74 0.20 0.16 0.85"),   # thin tall cyl
    ("box",      "0.035 0.035 0.35",  "0.70 0.18 0.18 0.85"),   # medium pillar
    # --- Wide walls (hard to fly around) ---
    ("box",      "0.12 0.018 0.30",   "0.68 0.20 0.18 0.80"),   # tall thin wall
    ("box",      "0.10 0.015 0.25",   "0.76 0.22 0.14 0.80"),   # shorter wide wall
    # --- Floating / mid-air obstacles (placed at flight altitude) ---
    ("sphere",   "0.07",              "0.75 0.22 0.12 0.70"),   # floating sphere
    ("box",      "0.08 0.08 0.04",    "0.70 0.18 0.20 0.70"),   # floating slab
    ("cylinder", "0.06 0.035",        "0.70 0.16 0.20 0.70"),   # floating disc
    ("sphere",   "0.055",             "0.72 0.24 0.18 0.70"),   # small floating sphere
]


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
        # -- obstacle / rangefinder --
        n_obstacle_slots: int = 12,
        n_obstacles_range: tuple = (4, 10),
        n_blocker_obstacles: int = 2,
        obstacle_goal_clearance: float = 0.0,
        obstacle_z_extra_range: float = 0.10,
        rangefinder_mode: str = "lidar",
        n_lidar_rays: int = 16,
        rangefinder_max_range: float = 2.0,
        rangefinder_noise_std: float = 0.02,
        obstacle_safety_margin: float = 0.15,
        obstacle_collision_penalty: float = 5.0,
        obstacle_proximity_coeff: float = 0.3,
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
            n_obstacle_slots=n_obstacle_slots,
            n_obstacles_range=n_obstacles_range,
            n_blocker_obstacles=n_blocker_obstacles,
            obstacle_goal_clearance=obstacle_goal_clearance,
            obstacle_z_extra_range=obstacle_z_extra_range,
            rangefinder_mode=rangefinder_mode,
            n_lidar_rays=n_lidar_rays,
            rangefinder_max_range=rangefinder_max_range,
            rangefinder_noise_std=rangefinder_noise_std,
            obstacle_safety_margin=obstacle_safety_margin,
            obstacle_collision_penalty=obstacle_collision_penalty,
            obstacle_proximity_coeff=obstacle_proximity_coeff,
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
    """Motor-level Crazyflie 2.1 reach task with random obstacles.

    Inherits **all** physics, motor dynamics, aerodynamics, battery
    modelling, and rendering infrastructure from
    :class:`CrazyflieReachEnv`.  This class adds:

    * **Random obstacles** — mocap bodies with collision geoms injected
      into the XML.  At each :meth:`reset` a random subset is activated
      and placed in the workspace with rejection-sampling constraints.
    * **Multi-ray rangefinder** — ``mj_ray``-based distance sensor with
      configurable ray count, max range, and Gaussian noise.
    * **Modified reward** — parent reach reward plus continuous obstacle
      proximity penalty and hard collision termination.
    * **Extended observations** — parent obs concatenated with normalised
      rangefinder readings (0 = touching, 1 = max range / clear).
    """

    def __init__(
        self,
        # ---- obstacle parameters ----
        n_obstacle_slots: int = 12,
        n_obstacles_range: Tuple[int, int] = (4, 10),
        obstacle_min_sep: float = 0.10,
        obstacle_spawn_clearance: float = 0.15,
        obstacle_goal_clearance: float = 0.0,
        obstacle_z_extra_range: float = 0.10,
        n_blocker_obstacles: int = 2,
        # ---- rangefinder parameters ----
        rangefinder_mode: str = "lidar",
        n_lidar_rays: int = 16,
        rangefinder_max_range: float = 2.0,
        rangefinder_noise_std: float = 0.02,
        # ---- reward tuning ----
        obstacle_safety_margin: float = 0.15,
        obstacle_collision_penalty: float = 5.0,
        obstacle_proximity_coeff: float = 0.3,
        **kwargs,
    ) -> None:
        # ---- Store BEFORE super().__init__ (which calls _load_model_xml) --
        self._n_obstacle_slots = int(max(0, n_obstacle_slots))
        self._n_obstacles_range = (
            int(max(0, n_obstacles_range[0])),
            int(max(n_obstacles_range[0], n_obstacles_range[1])),
        )
        self._obstacle_min_sep = float(max(0, obstacle_min_sep))
        self._obstacle_spawn_clearance = float(max(0, obstacle_spawn_clearance))
        self._obstacle_goal_clearance = float(max(0, obstacle_goal_clearance))
        self._obstacle_z_extra = float(max(0, obstacle_z_extra_range))
        self._n_blocker_obstacles = int(max(0, n_blocker_obstacles))

        self._rangefinder_mode = str(rangefinder_mode)
        self._n_lidar_rays = int(max(1, n_lidar_rays))
        self._rangefinder_max_range = float(max(0.1, rangefinder_max_range))
        self._rangefinder_noise_std = float(max(0, rangefinder_noise_std))

        self._obstacle_safety_margin = float(max(0.01, obstacle_safety_margin))
        self._obstacle_collision_penalty = float(max(0, obstacle_collision_penalty))
        self._obstacle_proximity_coeff = float(max(0, obstacle_proximity_coeff))

        # ---- Build model via parent (calls overridden _load_model_xml) ----
        super().__init__(**kwargs)

        # ---- Post-init: resolve obstacle body / geom / mocap IDs ---------
        self._obstacle_body_ids: list[int] = []
        self._obstacle_mocap_ids: list[int] = []
        self._obstacle_geom_ids: set[int] = set()
        self._obstacle_half_heights: list[float] = []

        for i in range(self._n_obstacle_slots):
            bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"obs_{i}",
            )
            gid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_{i}_geom",
            )
            if bid < 0 or gid < 0:
                continue
            self._obstacle_body_ids.append(bid)
            self._obstacle_mocap_ids.append(int(self.model.body_mocapid[bid]))
            self._obstacle_geom_ids.add(gid)

            # Compute vertical half-extent for ground-level placement
            gtype = int(self.model.geom_type[gid])
            gsize = self.model.geom_size[gid]
            if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                hh = float(gsize[2])
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                hh = float(gsize[1])
            elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                hh = float(gsize[0])
            else:
                hh = 0.10
            self._obstacle_half_heights.append(hh)

        # ---- Rangefinder setup -------------------------------------------
        self._rf_dirs_body = self._build_ray_directions()
        self._n_rf_total = len(self._rf_dirs_body)

        # Geom-group mask: only test group 1 (obstacles + floor + walls)
        self._rf_geomgroup = np.zeros(6, dtype=np.uint8)
        self._rf_geomgroup[1] = 1

        # ---- Active obstacle state (updated each reset) ------------------
        self._n_active_obstacles = 0
        self._active_obstacle_positions = np.zeros(
            (self._n_obstacle_slots, 3), dtype=np.float64,
        )

    # ==================================================================
    #  XML injection
    # ==================================================================
    def _load_model_xml(self, model_path: str) -> str:
        """Override: inject obstacles and assign geom groups for rangefinder."""
        xml_str = super()._load_model_xml(model_path)
        root = ET.fromstring(xml_str)

        worldbody = root.find("worldbody")
        if worldbody is None:
            return xml_str

        # --- Push cameras further out so obstacles don't block the view ---
        #     The base XML cameras use mode="trackcom" with a close offset
        #     (0.55 m side, 1.2 m top) — obstacles easily occlude them.
        #     Switch to fixed birds-eye cameras that clear the obstacle field.
        _cam_overrides = {
            "cf_side": {
                "mode": "fixed",
                "pos": "1.6 -0.6 0.9",
                "xyaxes": "0.4 0.92 0 -0.15 0.06 0.98",
                "fovy": "60",
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

        # --- Move floor & arena walls to geom group 1 (rangefinder-visible) ---
        for geom in worldbody.findall("geom"):
            name = geom.get("name", "")
            if name in (
                "floor", "arena_north", "arena_south",
                "arena_east", "arena_west",
            ):
                geom.set("group", "1")

        # --- Inject N obstacle mocap bodies (initially underground) --------
        for i in range(self._n_obstacle_slots):
            shape_type, shape_size, shape_rgba = (
                _OBSTACLE_DEFS[i % len(_OBSTACLE_DEFS)]
            )
            obs_body = ET.SubElement(
                worldbody, "body",
                name=f"obs_{i}", mocap="true", pos="0 0 -5",
            )
            ET.SubElement(
                obs_body, "geom",
                name=f"obs_{i}_geom",
                type=shape_type,
                size=shape_size,
                rgba=shape_rgba,
                contype="1", conaffinity="1",
                group="1",
                # Friction matching the drone so contacts feel right
                friction="0.9 0.05 0.02",
                solref="0.005 1",
                solimp="0.9 0.95 0.001",
            )

        return ET.tostring(root, encoding="unicode")

    # ==================================================================
    #  Rangefinder
    # ==================================================================
    def _build_ray_directions(self) -> np.ndarray:
        """Unit-length ray directions in **body frame**."""
        dirs: list[list[float]] = []

        if self._rangefinder_mode == "multi_ranger":
            # Crazyflie Multi-ranger deck: 5 VL53L1x ToF sensors
            dirs = [
                [1, 0, 0],     # front  (+X body)
                [-1, 0, 0],    # back
                [0, 1, 0],     # left   (+Y body)
                [0, -1, 0],    # right
                [0, 0, 1],     # up
            ]
        else:
            # "lidar" — horizontal ring + vertical pair
            for i in range(self._n_lidar_rays):
                angle = 2.0 * math.pi * i / self._n_lidar_rays
                dirs.append([math.cos(angle), math.sin(angle), 0.0])
            dirs.append([0.0, 0.0, 1.0])    # up
            dirs.append([0.0, 0.0, -1.0])   # down

        arr = np.array(dirs, dtype=np.float64)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-8)

    def _compute_rangefinder(self) -> np.ndarray:
        """Cast rays from the drone body; return normalised distances [0, 1].

        0 → surface contact, 1 → max range (or no hit).
        """
        pos = self.data.xpos[self.cf_body_id].copy()
        rot = self.data.xmat[self.cf_body_id].reshape(3, 3)
        geomid_out = np.array([-1], dtype=np.int32)

        readings = np.ones(self._n_rf_total, dtype=np.float32)

        for i, dir_body in enumerate(self._rf_dirs_body):
            dir_world = (rot @ dir_body).astype(np.float64)
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
        """Return True if any drone geom is in contact with an obstacle geom."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (
                (g1 in self._drone_geom_ids and g2 in self._obstacle_geom_ids)
                or (g2 in self._drone_geom_ids and g1 in self._obstacle_geom_ids)
            ):
                return True
        return False

    # ==================================================================
    #  Obstacle placement
    # ==================================================================
    def _reset_obstacles(self) -> None:
        """Activate a random subset and place them.

        Placement strategy:
        1. **Blocker obstacles** — deliberately placed on the line between
           spawn and goal so the drone *must* navigate around them.
        2. **Random obstacles** — scattered across the workspace with
           rejection sampling (ground-standing or floating at flight alt).
        """
        n_active = int(
            self._rng.integers(
                self._n_obstacles_range[0],
                self._n_obstacles_range[1] + 1,
            )
        )
        n_active = min(n_active, len(self._obstacle_mocap_ids))
        self._n_active_obstacles = n_active

        placed: list[np.ndarray] = []  # XY of already-placed obstacles

        # --- Phase 1: place blocker obstacles on the spawn→goal corridor ---
        n_blockers = min(self._n_blocker_obstacles, n_active)
        for i in range(n_blockers):
            pos = self._sample_blocker_pos(i, placed)
            mid = self._obstacle_mocap_ids[i]
            self.data.mocap_pos[mid] = pos
            self._active_obstacle_positions[i] = pos
            placed.append(pos[:2])

        # --- Phase 2: place remaining obstacles randomly ------------------
        for i in range(n_blockers, n_active):
            pos = self._sample_obstacle_pos(i, placed)
            mid = self._obstacle_mocap_ids[i]
            self.data.mocap_pos[mid] = pos
            self._active_obstacle_positions[i] = pos
            placed.append(pos[:2])

        # Deactivate remaining slots (underground)
        for i in range(n_active, len(self._obstacle_mocap_ids)):
            mid = self._obstacle_mocap_ids[i]
            self.data.mocap_pos[mid] = [0.0, 0.0, -5.0]
            self._active_obstacle_positions[i] = [0.0, 0.0, -5.0]

    def _sample_blocker_pos(
        self,
        idx: int,
        placed: list[np.ndarray],
    ) -> np.ndarray:
        """Place an obstacle on the spawn → goal corridor.

        The obstacle is put at a random fraction (0.25–0.75) along the
        line from spawn to goal, with a small lateral jitter so the
        drone can't just memorise a single dodge direction.
        """
        hh = (
            self._obstacle_half_heights[idx]
            if idx < len(self._obstacle_half_heights)
            else 0.10
        )
        spawn_xy = self.spawn_pos[:2]
        goal_xy = self._goal_pos[:2]
        delta = goal_xy - spawn_xy
        perp = np.array([-delta[1], delta[0]], dtype=float)
        perp_len = float(np.linalg.norm(perp))
        if perp_len > 1e-6:
            perp /= perp_len

        for _ in range(60):
            t = float(self._rng.uniform(0.25, 0.75))
            mid_xy = spawn_xy + t * delta
            # Lateral offset ±0.08 m so it's not perfectly centred
            lateral = float(self._rng.uniform(-0.08, 0.08))
            xy = mid_xy + lateral * perp
            xy = np.clip(xy, -self.goal_xy_range * 0.95, self.goal_xy_range * 0.95)

            # Still respect spawn clearance
            if np.linalg.norm(xy - spawn_xy) < self._obstacle_spawn_clearance:
                continue
            # Not on top of another placed obstacle
            ok = True
            for prev_xy in placed:
                if np.linalg.norm(xy - prev_xy) < self._obstacle_min_sep:
                    ok = False
                    break
            if not ok:
                continue

            # Ground-based: bottom at floor level so it blocks the full column
            z = hh
            return np.array([float(xy[0]), float(xy[1]), z], dtype=np.float64)

        return np.array([0.0, 0.0, -5.0], dtype=np.float64)

    def _sample_obstacle_pos(
        self,
        idx: int,
        placed: list[np.ndarray],
    ) -> np.ndarray:
        """Place an obstacle randomly — either ground-standing or floating."""
        hh = (
            self._obstacle_half_heights[idx]
            if idx < len(self._obstacle_half_heights)
            else 0.10
        )
        # Floating obstacles (idx 8-11 in the palette) get mid-air placement
        is_floating = (idx % len(_OBSTACLE_DEFS)) >= 8

        for _ in range(100):
            x = float(
                self._rng.uniform(
                    -self.goal_xy_range * 0.95,
                    self.goal_xy_range * 0.95,
                )
            )
            y = float(
                self._rng.uniform(
                    -self.goal_xy_range * 0.95,
                    self.goal_xy_range * 0.95,
                )
            )
            xy = np.array([x, y])

            # Too close to spawn?
            if np.linalg.norm(xy - self.spawn_pos[:2]) < self._obstacle_spawn_clearance:
                continue
            # Too close to current goal centre?
            if self._obstacle_goal_clearance > 0:
                if np.linalg.norm(xy - self._goal_pos[:2]) < self._obstacle_goal_clearance:
                    continue
            # Too close to another obstacle?
            ok = True
            for prev_xy in placed:
                if np.linalg.norm(xy - prev_xy) < self._obstacle_min_sep:
                    ok = False
                    break
            if not ok:
                continue

            if is_floating:
                # Float at typical flight altitude (0.20 – 0.60 m)
                z = hh + float(self._rng.uniform(0.15, 0.55))
            else:
                # Ground-based: bottom on floor, optional small z jitter
                z = hh + float(self._rng.uniform(0.0, self._obstacle_z_extra))
            return np.array([x, y, z], dtype=np.float64)

        # Fallback: underground (doesn't count as active)
        return np.array([0.0, 0.0, -5.0], dtype=np.float64)

    # ==================================================================
    #  Overridden API
    # ==================================================================
    @property
    def observation_dim(self) -> int:
        return super().observation_dim + self._n_rf_total

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        # Parent reset: spawns drone, samples goal, clears odom, etc.
        super().reset(seed=seed)
        # Place obstacles (needs goal_pos from parent reset)
        self._reset_obstacles()
        mujoco.mj_forward(self.model, self.data)
        return self._observe()

    def _observe(self) -> np.ndarray:
        base_obs = super()._observe()
        rf = self._compute_rangefinder()
        return np.concatenate([base_obs, rf]).astype(np.float32)

    def _compute_reward_and_info(self):
        reward, done, info = super()._compute_reward_and_info()

        # ---- Rangefinder-based proximity penalty ----
        rf = self._compute_rangefinder()
        min_rf_norm = float(np.min(rf))
        min_range_m = min_rf_norm * self._rangefinder_max_range

        info["min_obstacle_range_m"] = min_range_m
        info["rangefinder_min_norm"] = min_rf_norm

        if min_range_m < self._obstacle_safety_margin:
            frac = 1.0 - min_range_m / self._obstacle_safety_margin
            penalty = -self._obstacle_proximity_coeff * frac * frac
            reward += penalty
            info["obstacle_proximity_penalty"] = float(penalty)
        else:
            info["obstacle_proximity_penalty"] = 0.0

        # ---- Hard collision termination ----
        if self._check_obstacle_collision():
            reward -= self._obstacle_collision_penalty
            done = True
            info["obstacle_collision"] = True
        else:
            info["obstacle_collision"] = False

        return float(reward), done, info

    # ==================================================================
    #  Minimap override (show obstacles as red dots)
    # ==================================================================
    def _draw_minimap(self, shape: tuple) -> np.ndarray:
        panel = super()._draw_minimap(shape)
        h, w = shape

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

        # Draw active obstacles
        oc = (200, 60, 60)
        oc_ring = (240, 90, 80)
        for i in range(self._n_active_obstacles):
            opos = self._active_obstacle_positions[i]
            if opos[2] < -1.0:
                continue
            ox, oy = _w2p(float(opos[0]), float(opos[1]))
            # Filled circle (r≈5)
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    if dx * dx + dy * dy <= 25:
                        yy, xx = oy + dy, ox + dx
                        if 0 <= yy < h and 0 <= xx < w:
                            panel[yy, xx] = oc
            # Outer ring
            for a_i in range(24):
                ang = a_i * 2.0 * math.pi / 24
                rx = ox + int(7 * math.cos(ang))
                ry = oy + int(7 * math.sin(ang))
                if 0 <= ry < h and 0 <= rx < w:
                    panel[ry, rx] = oc_ring

        return panel
