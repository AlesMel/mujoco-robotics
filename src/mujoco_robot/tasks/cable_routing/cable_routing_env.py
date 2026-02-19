"""UR3e cable-routing task with an anchored deformable cable.

This environment targets fixture/clip routing with one cable endpoint already
connected to a board anchor. The robot grasps the free cable end with suction
and routes it through a sequence of clips.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import gymnasium
import mujoco
import numpy as np

from mujoco_robot.assets.actuators import (
    configure_position_actuators,
    resolve_robot_actuators,
)
from mujoco_robot.assets.configs import get_robot_config
from mujoco_robot.core.xml_builder import (
    ensure_spawn_table,
    inject_side_camera,
    load_robot_xml,
    set_framebuffer_size,
)
from mujoco_robot.envs.step_result import StepResult

_DEFAULT_MODEL = str(
    Path(__file__).resolve().parent.parent.parent / "assets" / "ur3e.xml"
)

_EPICK_BODY_MESH = (
    Path(__file__).resolve().parent.parent.parent
    / "assets"
    / "meshes"
    / "epick"
    / "epick_body.stl"
)
# Real Robotiq e-Pick specs:
#   Body length  : 102.3 mm   (unchanged)
#   Body diameter: ~43 mm   → radius 21.5 mm
#   Suction-cup face diameter: ~32 mm → radius 16 mm
_EPICK_BODY_LENGTH = 0.1023
_EPICK_BODY_COLLISION_RADIUS = 0.022   # 43 mm outer diameter
_EPICK_CUP_RADIUS = 0.016             # 32 mm suction-face diameter
_EPICK_CUP_HEIGHT = 0.015
# Robotiq e-Pick mass (datasheet: 235 g)
_EPICK_MASS = 0.235
# Approximate principal inertia for a cylinder M=0.235 kg, R=0.022 m, H=0.1023 m
_EPICK_INERTIA_XX = 0.00027   # kg·m²  (transverse)
_EPICK_INERTIA_ZZ = 0.000057  # kg·m²  (axial)

_FALLBACK_ROBOT_XML: str = ""
try:
    _FALLBACK_ROBOT_XML = Path(_DEFAULT_MODEL).read_text()
except FileNotFoundError:
    pass


class CableRoutingGymnasium(gymnasium.Env):
    """Gymnasium wrapper around :class:`URCableRoutingEnv`."""

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        seed: int | None = None,
        render: bool = False,
        render_mode: str | None = None,
        time_limit: int = 450,
        model_path: str = _DEFAULT_MODEL,
        actuator_profile: str = "ur3e",
        **env_kwargs,
    ):
        resolved_render_mode = render_mode or ("rgb_array" if render else None)
        if resolved_render_mode not in {None, "rgb_array", "human"}:
            raise ValueError("render_mode must be one of: None, 'rgb_array', 'human'")

        self.base = URCableRoutingEnv(
            model_path=model_path,
            actuator_profile=actuator_profile,
            render_size=(640, 480) if resolved_render_mode == "rgb_array" else (160, 120),
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
        self.render_mode = resolved_render_mode
        self._human_viewer = None

    def reset(self, *, seed: int | None = None, options=None):
        obs = self.base.reset(seed=seed)
        if self.render_mode == "human":
            self.render()
        return obs.astype(np.float32), {}

    def step(self, action):
        res: StepResult = self.base.step(action)
        terminated = bool(res.info.get("success", False))
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
                except Exception as exc:  # pragma: no cover - GUI availability depends on host.
                    raise RuntimeError(
                        "Human rendering requires mujoco.viewer with GUI support."
                    ) from exc
                self._human_viewer = mj_viewer.launch_passive(self.base.model, self.base.data)

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


class URCableRoutingEnv:
    """Single-arm cable-routing environment with sequential clip objectives."""

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        actuator_profile: str = "ur3e",
        time_limit: int = 450,
        ee_step: float = 0.04,
        yaw_step: float = 0.45,
        suction_radius: float = 0.025,
        suction_alignment_cos_min: float = 0.55,
        suction_detach_dist_scale: float = 1.6,
        suction_attach_stochastic: bool = True,
        suction_success_prob_floor: float = 0.20,
        suction_success_prob_ceiling: float = 0.99,
        suction_attach_fail_penalty: float = 0.01,
        adhesion_gain: float = 35.0,
        clip_capture_radius: float = 0.014,
        clip_capture_z_tol: float = 0.018,
        clip_hold_steps_required: int = 5,
        clip_progress_bonus: float = 1.5,
        completion_bonus: float = 6.0,
        cable_points: int = 24,
        cable_radius: float = 0.005,
        cable_joint_damping: float = 1.5,
        cable_length: float = 0.24,
        clip_positions: Sequence[Sequence[float]] | None = None,
        settle_steps: int = 500,
        zero_cable_velocity_on_reset: bool = True,
        render_size: Tuple[int, int] = (960, 720),
        seed: Optional[int] = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)

        self.model_path = model_path
        self.actuator_profile = actuator_profile
        self.time_limit = int(time_limit)
        self.ee_step = float(ee_step)
        self.yaw_step = float(yaw_step)
        self.suction_radius = float(max(1e-4, suction_radius))
        self.suction_alignment_cos_min = float(
            np.clip(suction_alignment_cos_min, 0.0, 0.999)
        )
        self.suction_detach_dist_scale = float(max(1.0, suction_detach_dist_scale))
        self.suction_attach_stochastic = bool(suction_attach_stochastic)
        self.suction_success_prob_floor = float(np.clip(suction_success_prob_floor, 0.0, 1.0))
        self.suction_success_prob_ceiling = float(
            np.clip(
                max(suction_success_prob_ceiling, self.suction_success_prob_floor),
                0.0,
                1.0,
            )
        )
        self.suction_attach_fail_penalty = float(max(0.0, suction_attach_fail_penalty))
        self.adhesion_gain = float(max(0.0, adhesion_gain))

        self.clip_capture_radius = float(max(1e-4, clip_capture_radius))
        self.clip_capture_z_tol = float(max(1e-4, clip_capture_z_tol))
        self.clip_hold_steps_required = int(max(1, clip_hold_steps_required))
        self.clip_progress_bonus = float(max(0.0, clip_progress_bonus))
        self.completion_bonus = float(max(0.0, completion_bonus))

        self.cable_points = int(max(4, cable_points))
        self.cable_radius = float(max(1e-4, cable_radius))
        self.cable_joint_damping = float(max(0.0, cable_joint_damping))
        self.cable_length = float(max(0.08, cable_length))
        self.zero_cable_velocity_on_reset = bool(zero_cable_velocity_on_reset)

        self.render_size = render_size
        self.settle_steps = int(max(0, settle_steps))

        # Control / stability settings.
        self.max_joint_vel = 4.0
        self.ik_damping = 0.02
        self.ik_gain = 0.15
        self.hold_eps = 0.02
        self.n_substeps = 5
        self.init_q = np.array(
            [-0.4147, -math.pi / 2.0, math.pi / 2.0, -math.pi / 2.0, -math.pi / 2.0, 0.0],
            dtype=float,
        )
        self.ee_bounds = np.array(
            [[-0.65, 0.35], [-0.50, 0.50], [0.60, 1.45]],
            dtype=float,
        )

        # Board / clip defaults (world frame).
        self.board_center = np.array([0.16, 0.0, 0.746], dtype=float)
        self.board_half = np.array([0.13, 0.10, 0.006], dtype=float)
        self.board_top_z = float(self.board_center[2] + self.board_half[2])

        default_clip_xy = np.array(
            [
                [0.08, -0.065],
                [0.12, 0.050],
                [0.18, -0.045],
                [0.24, 0.055],
            ],
            dtype=float,
        )
        if clip_positions is None:
            clip_positions_arr = np.column_stack(
                [
                    default_clip_xy,
                    np.full(default_clip_xy.shape[0], self.board_top_z + 0.014, dtype=float),
                ]
            )
        else:
            pts = np.asarray(clip_positions, dtype=float)
            if pts.ndim != 2 or pts.shape[1] not in {2, 3}:
                raise ValueError("clip_positions must be shape (N, 2) or (N, 3).")
            if pts.shape[1] == 2:
                clip_positions_arr = np.column_stack(
                    [
                        pts,
                        np.full(pts.shape[0], self.board_top_z + 0.014, dtype=float),
                    ]
                )
            else:
                clip_positions_arr = pts.copy()

        if clip_positions_arr.shape[0] < 1:
            raise ValueError("At least one clip target is required.")
        self._clip_positions = clip_positions_arr.astype(float)
        self.n_clips = int(self._clip_positions.shape[0])

        # Anchor at rear-left of the board, cable laid initially toward +x.
        self.anchor_position = np.array(
            [
                self.board_center[0] - self.board_half[0] + 0.015,
                self.board_center[1] - self.board_half[1] + 0.025,
                self.board_top_z + 0.018,
            ],
            dtype=float,
        )
        self.cable_initial_end = np.array(
            [
                min(self.board_center[0] + self.board_half[0] - 0.01, self.anchor_position[0] + self.cable_length),
                self.anchor_position[1] + 0.05,
                self.anchor_position[2] + 0.006,
            ],
            dtype=float,
        )

        # Align robot defaults with named robot configs when available.
        robot_name = actuator_profile
        if robot_name not in {"ur3e", "ur5e"}:
            stem = Path(model_path).stem.lower()
            if stem in {"ur3e", "ur5e"}:
                robot_name = stem
        try:
            robot_cfg = get_robot_config(robot_name)
            self.init_q = robot_cfg.init_q.copy()
            self.ee_bounds = robot_cfg.ee_bounds.copy()
        except ValueError:
            pass

        self._cable_prefix = "route_cable_"

        robot_xml = self._load_robot_xml(model_path)
        self.model_xml = self._build_env_xml(robot_xml)
        self.model = mujoco.MjModel.from_xml_string(self.model_xml)
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = 0.002
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        # Cable contacts benefit from more constraint solver iterations.
        self.model.opt.iterations = max(self.model.opt.iterations, 100)
        self.model.opt.noslip_iterations = max(self.model.opt.noslip_iterations, 20)
        self.model.opt.gravity[:] = np.array([0.0, 0.0, -9.81])

        self._camera_overview = self._resolve_camera_name(("route_overview", "top"))
        self._camera_board = self._resolve_camera_name(("route_board", "top"))
        self._camera_side = self._resolve_camera_name(("route_side", "side"))
        self._camera_ee = self._resolve_camera_name(("ee_cam",))

        rw, rh = self.render_size
        self._renderer_overview = mujoco.Renderer(self.model, height=rh, width=rw)
        self._renderer_board = mujoco.Renderer(self.model, height=rh, width=rw)
        self._renderer_side = mujoco.Renderer(self.model, height=rh, width=rw)
        self._renderer_ee = mujoco.Renderer(self.model, height=rh, width=rw)

        self.ee_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.suction_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "suction_site"
        )
        if self.suction_site < 0:
            self.suction_site = self.ee_site

        self.anchor_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "cable_anchor_site"
        )
        if self.anchor_site < 0:
            raise ValueError("Cable-routing MJCF is missing required 'cable_anchor_site'.")

        self.suction_adhesion_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "suction_adhesion"
        )
        if self.suction_adhesion_actuator_id < 0:
            raise ValueError("Cable-routing MJCF is missing required 'suction_adhesion' actuator.")

        self._clip_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"clip_target_{i}")
            for i in range(self.n_clips)
        ]
        if any(cid < 0 for cid in self._clip_site_ids):
            raise ValueError("Cable-routing MJCF is missing one or more clip target sites.")

        actuator_handles = resolve_robot_actuators(self.model, self.actuator_profile)
        self.robot_joints = list(actuator_handles.joint_names)
        self.robot_actuators = list(actuator_handles.actuator_ids)
        self.robot_dofs = list(actuator_handles.dof_ids)
        self._robot_joint_ids = list(actuator_handles.joint_ids)
        self._robot_qpos_ids = list(actuator_handles.qpos_ids)

        configure_position_actuators(
            self.model,
            actuator_handles,
            min_damping=16.0,
            min_frictionloss=1.0,
            kp=120.0,
        )

        self._cable_body_ids = self._collect_named_ids(
            mujoco.mjtObj.mjOBJ_BODY,
            self.model.nbody,
            self._cable_prefix,
        )
        self._cable_joint_ids = self._collect_named_ids(
            mujoco.mjtObj.mjOBJ_JOINT,
            self.model.njnt,
            self._cable_prefix,
        )
        if not self._cable_body_ids:
            raise ValueError(
                "No cable composite bodies found. Check composite prefix/name generation."
            )

        self._free_end_body_id = -1
        self._anchor_end_body_id = -1
        self._resolve_cable_endpoints_from_names()

        self.step_id = 0
        self.suction_on = False
        self.grasped = False
        self._target_yaw = 0.0
        self._last_targets = self.init_q.copy()
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._last_reward = 0.0
        self._last_step_info: Dict[str, float | bool | int] = {}

        self.routed_clip_count = 0
        self._clip_hold_steps = 0
        self._prev_target_planar_dist = 0.0

        self._last_attach_attempted = False
        self._last_attach_success = False
        self._last_attach_eligible = False
        self._last_attach_prob = 0.0
        self._last_attach_quality = 0.0
        self._last_progress_advanced = False

    # ------------------------------------------------------------------ XML
    def _load_robot_xml(self, path: str) -> str:
        try:
            return load_robot_xml(path)
        except FileNotFoundError:
            if _FALLBACK_ROBOT_XML:
                return _FALLBACK_ROBOT_XML
            raise FileNotFoundError(f"Robot MJCF not found at '{path}'.")

    def _composite_vertex_string(self) -> str:
        vertices: list[str] = []
        # Composite vertices are expressed in the parent body frame.
        local_end = self.cable_initial_end - self.anchor_position
        for i in range(self.cable_points):
            t = i / max(self.cable_points - 1, 1)
            p = t * local_end
            p[1] += 0.015 * math.sin(math.pi * t)
            p[2] -= 0.010 * math.sin(math.pi * t)
            vertices.extend([f"{p[0]:.5f}", f"{p[1]:.5f}", f"{p[2]:.5f}"])
        return " ".join(vertices)

    def _build_env_xml(self, robot_xml: str) -> str:
        root = ET.fromstring(robot_xml)

        set_framebuffer_size(root, self.render_size[0], self.render_size[1])
        ensure_spawn_table(root)
        inject_side_camera(root)
        self._inject_task_cameras(root)

        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        if _EPICK_BODY_MESH.exists():
            ET.SubElement(
                asset,
                "mesh",
                {
                    "name": "epick_body_mesh",
                    "file": str(_EPICK_BODY_MESH.resolve()),
                    "scale": "0.001 0.001 0.001",
                },
            )

        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("Robot MJCF missing <worldbody>")

        wrist3 = root.find(".//body[@name='wrist3']")
        if wrist3 is None:
            raise ValueError("Robot MJCF missing wrist3 body for suction tool attachment.")

        epick_base = ET.SubElement(
            wrist3,
            "body",
            {
                "name": "epick_base_link",
                "pos": "0 0.0921 0",
                "quat": "0.70710678 -0.70710678 0 0",
            },
        )
        # Give the e-Pick gripper its real mass (235 g) and inertia so that
        # the robot arm controller sees realistic load dynamics.
        ET.SubElement(
            epick_base,
            "inertial",
            {
                "mass": f"{_EPICK_MASS:.4f}",
                "pos": f"0 0 {_EPICK_BODY_LENGTH / 2.0:.6f}",
                "diaginertia": (
                    f"{_EPICK_INERTIA_XX:.6f} "
                    f"{_EPICK_INERTIA_XX:.6f} "
                    f"{_EPICK_INERTIA_ZZ:.6f}"
                ),
            },
        )
        epick_body = ET.SubElement(
            epick_base,
            "body",
            {
                "name": "robotiq_epick_body",
                "pos": f"0 0 {_EPICK_BODY_LENGTH / 2.0:.6f}",
            },
        )
        if _EPICK_BODY_MESH.exists():
            ET.SubElement(
                epick_body,
                "geom",
                {
                    "name": "robotiq_epick_body_visual",
                    "type": "mesh",
                    "mesh": "epick_body_mesh",
                    "pos": f"0 0 {-_EPICK_BODY_LENGTH / 2.0:.6f}",
                    "rgba": "0.25 0.25 0.25 1",
                    "contype": "0",
                    "conaffinity": "0",
                    "mass": "0",
                },
            )
        ET.SubElement(
            epick_body,
            "geom",
            {
                "name": "robotiq_epick_body_collision",
                "type": "cylinder",
                "size": f"{_EPICK_BODY_COLLISION_RADIUS:.6f} {_EPICK_BODY_LENGTH / 2.0:.6f}",
                "rgba": "0.2 0.2 0.2 0.1",
                "contype": "1",
                "conaffinity": "1",
                "friction": "1.0 0.1 0.02",
                "mass": "0",
            },
        )
        suction_cup = ET.SubElement(
            epick_body,
            "body",
            {
                "name": "robotiq_epick_suction_cup",
                "pos": f"0 0 {(_EPICK_BODY_LENGTH + _EPICK_CUP_HEIGHT) / 2.0:.6f}",
            },
        )
        ET.SubElement(
            suction_cup,
            "geom",
            {
                "name": "robotiq_epick_suction_cup_collision",
                "type": "cylinder",
                "size": f"{_EPICK_CUP_RADIUS:.6f} {_EPICK_CUP_HEIGHT / 2.0:.6f}",
                "rgba": "0.02 0.02 0.02 1",
                "contype": "1",
                "conaffinity": "1",
                "friction": "1.2 0.1 0.02",
                "margin": "0.004",
                "gap": "0.0015",
                "mass": "0",
            },
        )
        ET.SubElement(
            suction_cup,
            "site",
            {
                "name": "suction_site",
                "pos": f"0 0 {_EPICK_CUP_HEIGHT / 2.0:.6f}",
                "size": "0.006",
                "rgba": "0.1 0.9 0.9 1",
            },
        )

        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "routing_board",
                "type": "box",
                "pos": (
                    f"{self.board_center[0]:.5f} "
                    f"{self.board_center[1]:.5f} "
                    f"{self.board_center[2]:.5f}"
                ),
                "size": (
                    f"{self.board_half[0]:.5f} "
                    f"{self.board_half[1]:.5f} "
                    f"{self.board_half[2]:.5f}"
                ),
                "rgba": "0.82 0.83 0.86 1",
                "friction": "1.0 0.04 0.005",
            },
        )

        clip_wall_half = np.array([0.006, 0.0015, 0.009], dtype=float)
        clip_gap = 0.016
        clip_floor_half = np.array([0.006, clip_gap / 2.0 + 0.0015, 0.0012], dtype=float)
        for i, clip in enumerate(self._clip_positions):
            clip_body = ET.SubElement(
                worldbody,
                "body",
                {
                    "name": f"clip_{i}_body",
                    "pos": f"{clip[0]:.5f} {clip[1]:.5f} {clip[2]:.5f}",
                },
            )
            ET.SubElement(
                clip_body,
                "geom",
                {
                    "name": f"clip_{i}_left",
                    "type": "box",
                    "pos": f"0 {clip_gap / 2.0 + clip_wall_half[1]:.5f} 0",
                    "size": (
                        f"{clip_wall_half[0]:.5f} "
                        f"{clip_wall_half[1]:.5f} "
                        f"{clip_wall_half[2]:.5f}"
                    ),
                    "rgba": "0.15 0.22 0.85 1",
                    "friction": "1.4 0.03 0.002",
                },
            )
            ET.SubElement(
                clip_body,
                "geom",
                {
                    "name": f"clip_{i}_right",
                    "type": "box",
                    "pos": f"0 {-clip_gap / 2.0 - clip_wall_half[1]:.5f} 0",
                    "size": (
                        f"{clip_wall_half[0]:.5f} "
                        f"{clip_wall_half[1]:.5f} "
                        f"{clip_wall_half[2]:.5f}"
                    ),
                    "rgba": "0.15 0.22 0.85 1",
                    "friction": "1.4 0.03 0.002",
                },
            )
            ET.SubElement(
                clip_body,
                "geom",
                {
                    "name": f"clip_{i}_floor",
                    "type": "box",
                    "pos": f"0 0 {-clip_wall_half[2] + clip_floor_half[2]:.5f}",
                    "size": (
                        f"{clip_floor_half[0]:.5f} "
                        f"{clip_floor_half[1]:.5f} "
                        f"{clip_floor_half[2]:.5f}"
                    ),
                    "rgba": "0.12 0.18 0.72 1",
                    "friction": "1.4 0.03 0.002",
                },
            )
            ET.SubElement(
                clip_body,
                "site",
                {
                    "name": f"clip_target_{i}",
                    "type": "sphere",
                    "size": "0.006",
                    "rgba": "0.1 0.95 0.2 0.7",
                },
            )

        anchor_body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": "cable_anchor",
                "pos": (
                    f"{self.anchor_position[0]:.5f} "
                    f"{self.anchor_position[1]:.5f} "
                    f"{self.anchor_position[2]:.5f}"
                ),
            },
        )
        ET.SubElement(
            anchor_body,
            "geom",
            {
                "name": "cable_anchor_geom",
                "type": "cylinder",
                "size": "0.008 0.012",
                "rgba": "0.20 0.20 0.20 1",
                "friction": "1.0 0.02 0.001",
                "contype": "0",
                "conaffinity": "0",
                "mass": "0.02",
            },
        )
        ET.SubElement(
            anchor_body,
            "site",
            {
                "name": "cable_anchor_site",
                "type": "sphere",
                "size": "0.005",
                "rgba": "0.95 0.7 0.15 0.9",
            },
        )

        cable_composite = ET.SubElement(
            anchor_body,
            "composite",
            {
                "prefix": self._cable_prefix,
                "type": "cable",
                "initial": "none",
                "vertex": self._composite_vertex_string(),
            },
        )
        ET.SubElement(
            cable_composite,
            "joint",
            {
                "kind": "main",
                "damping": f"{self.cable_joint_damping:.6f}",
                # Higher armature prevents near-singular inertia matrices that
                # cause velocity blow-up in stiff cable simulations.
                "armature": "0.001",
            },
        )
        ET.SubElement(
            cable_composite,
            "geom",
            {
                "type": "capsule",
                "size": f"{self.cable_radius:.6f}",
                "rgba": "0.92 0.55 0.22 1",
                "friction": "1.4 0.02 0.001",
                "density": "1000",
                "condim": "3",
                # solref[0] must be well above the timestep (0.002 s) to avoid
                # stiff-contact instability; 0.02 s is a safe margin.
                "solref": "0.02 1",
                "solimp": "0.95 0.99 0.001",
            },
        )
        ET.SubElement(
            cable_composite,
            "site",
            {
                "size": "0.0035",
                "rgba": "0.95 0.65 0.20 0.45",
            },
        )

        actuator = root.find("actuator")
        if actuator is None:
            actuator = ET.SubElement(root, "actuator")
        ET.SubElement(
            actuator,
            "adhesion",
            {
                "name": "suction_adhesion",
                "body": "robotiq_epick_suction_cup",
                "gain": f"{self.adhesion_gain:.6f}",
                "ctrlrange": "0 1",
            },
        )

        return ET.tostring(root, encoding="unicode")

    def _inject_task_cameras(self, root: ET.Element) -> None:
        worldbody = root.find("worldbody")
        if worldbody is None:
            return

        def _remove_camera(name: str) -> None:
            cam = worldbody.find(f"./camera[@name='{name}']")
            if cam is not None:
                worldbody.remove(cam)

        for name in ("route_overview", "route_board", "route_side"):
            _remove_camera(name)

        ET.SubElement(
            worldbody,
            "camera",
            {
                "name": "route_overview",
                "mode": "fixed",
                "pos": "0.18 -0.62 1.45",
                "xyaxes": "1 0 0 0 0.83 0.56",
                "fovy": "50",
            },
        )
        ET.SubElement(
            worldbody,
            "camera",
            {
                "name": "route_board",
                "mode": "fixed",
                "pos": "0.48 -0.34 1.08",
                "xyaxes": "0.88 0.47 0 -0.22 0.42 0.88",
                "fovy": "45",
            },
        )
        ET.SubElement(
            worldbody,
            "camera",
            {
                "name": "route_side",
                "mode": "fixed",
                "pos": "0.88 -0.10 1.12",
                "xyaxes": "0.20 0.98 0 -0.65 0.13 0.75",
                "fovy": "50",
            },
        )

    def _resolve_camera_name(self, candidates: Sequence[str]) -> str:
        for name in candidates:
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0:
                return name
        raise ValueError(f"No camera found among candidates: {tuple(candidates)}")

    # ------------------------------------------------------------------ Helpers
    def _collect_named_ids(
        self,
        obj_type: mujoco.mjtObj,
        count: int,
        prefix: str,
    ) -> list[int]:
        ids: list[int] = []
        for i in range(count):
            name = mujoco.mj_id2name(self.model, obj_type, i)
            if name is None:
                continue
            if name.startswith(prefix):
                ids.append(i)
        return ids

    def _resolve_cable_endpoints_from_names(self) -> None:
        indexed: list[tuple[tuple[int, ...], int]] = []
        for bid in self._cable_body_ids:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if not name:
                continue
            nums = tuple(int(v) for v in re.findall(r"\d+", name))
            if nums:
                indexed.append((nums, bid))

        if indexed:
            indexed.sort(key=lambda item: item[0])
            self._anchor_end_body_id = indexed[0][1]
            self._free_end_body_id = indexed[-1][1]
        else:
            self._resolve_cable_endpoints_from_distance()

    def _resolve_cable_endpoints_from_distance(self) -> None:
        anchor = self.data.site_xpos[self.anchor_site].copy()
        body_positions = np.array([self.data.xpos[bid] for bid in self._cable_body_ids])
        dists = np.linalg.norm(body_positions - anchor[None, :], axis=1)
        self._anchor_end_body_id = self._cable_body_ids[int(np.argmin(dists))]
        self._free_end_body_id = self._cable_body_ids[int(np.argmax(dists))]

    def _joint_dof_slice(self, joint_id: int) -> slice:
        start = int(self.model.jnt_dofadr[joint_id])
        if joint_id + 1 < self.model.njnt:
            end = int(self.model.jnt_dofadr[joint_id + 1])
        else:
            end = int(self.model.nv)
        return slice(start, end)

    def _zero_cable_joint_velocities(self) -> None:
        for joint_id in self._cable_joint_ids:
            dof_slice = self._joint_dof_slice(joint_id)
            if dof_slice.start < dof_slice.stop:
                self.data.qvel[dof_slice] = 0.0

    # ------------------------------------------------------------------ API
    @property
    def action_dim(self) -> int:
        """Action: ``[dx, dy, dz, dyaw, suction]``."""
        return 5

    @property
    def observation_dim(self) -> int:
        """Observation dimensionality."""
        return 38 + 3 * self.n_clips

    # ------------------------------------------------------------------ Reset
    def _table_top_z(self) -> float:
        table_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        if table_id >= 0:
            table_pos = self.model.geom_pos[table_id]
            table_size = self.model.geom_size[table_id]
            return float(table_pos[2] + table_size[2])
        return 0.74

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        self.step_id = 0
        self.suction_on = False
        self.grasped = False
        self._last_action[:] = 0.0
        self._last_reward = 0.0
        self._last_step_info = {}

        self.routed_clip_count = 0
        self._clip_hold_steps = 0
        self._last_progress_advanced = False

        self._last_attach_attempted = False
        self._last_attach_success = False
        self._last_attach_eligible = False
        self._last_attach_prob = 0.0
        self._last_attach_quality = 0.0

        for qi, (qpos_adr, dof_adr, act_id) in enumerate(
            zip(self._robot_qpos_ids, self.robot_dofs, self.robot_actuators)
        ):
            self.data.qpos[qpos_adr] = self.init_q[qi]
            self.data.qvel[dof_adr] = 0.0
            self.data.ctrl[act_id] = self.init_q[qi]

        self._last_targets = self.init_q.copy()
        self._set_suction_adhesion_ctrl(0.0)

        mujoco.mj_forward(self.model, self.data)
        self._target_yaw = self._ee_yaw()
        self._resolve_cable_endpoints_from_distance()

        for _ in range(self.settle_steps):
            for k, act_id in enumerate(self.robot_actuators):
                self.data.ctrl[act_id] = self._last_targets[k]
            mujoco.mj_step(self.model, self.data)

        if self.zero_cable_velocity_on_reset:
            self._zero_cable_joint_velocities()
            mujoco.mj_forward(self.model, self.data)

        self._resolve_cable_endpoints_from_distance()
        self._prev_target_planar_dist = self._routing_metrics()["target_planar_dist"]
        return self._observe()

    # ------------------------------------------------------------------ IK + control
    def _ee_yaw(self) -> float:
        mat = self.data.site_xmat[self.ee_site].reshape(3, 3)
        return float(math.atan2(mat[1, 0], mat[0, 0]))

    def _suction_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.suction_site].copy()

    def _anchor_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.anchor_site].copy()

    def _free_end_pos(self) -> np.ndarray:
        return self.data.xpos[self._free_end_body_id].copy()

    def _free_end_vel(self) -> np.ndarray:
        return self.data.cvel[self._free_end_body_id, 3:6].copy()

    def _desired_ee(self, delta_xyz: np.ndarray, dyaw: float) -> Tuple[np.ndarray, float]:
        pos = self.data.site_xpos[self.ee_site].copy() + delta_xyz
        lo = self.ee_bounds[:, 0]
        hi = self.ee_bounds[:, 1]
        pos[0] = float(np.clip(pos[0], lo[0], hi[0]))
        pos[1] = float(np.clip(pos[1], lo[1], hi[1]))
        pos[2] = float(np.clip(pos[2], max(lo[2], self._table_top_z() + 0.03), hi[2]))
        self._target_yaw = self._target_yaw + float(dyaw)
        return pos, self._target_yaw

    def _ik_cartesian(self, target_pos: np.ndarray, target_yaw: float) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site)

        pos_err = target_pos - self.data.site_xpos[self.ee_site]
        yaw_err = target_yaw - self._ee_yaw()
        yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
        target = np.concatenate([pos_err, np.array([yaw_err], dtype=float)])

        cols = self.robot_dofs
        jmat = np.vstack([jacp[:, cols], jacr[2:3, cols]])
        lam = float(self.ik_damping)
        jj = jmat @ jmat.T + (lam * lam) * np.eye(jmat.shape[0])
        return jmat.T @ np.linalg.solve(jj, target)

    def _clamp_to_limits(self, targets: np.ndarray) -> np.ndarray:
        out = targets.copy()
        for k, jid in enumerate(self._robot_joint_ids):
            lo, hi = self.model.jnt_range[jid]
            if lo < hi:
                out[k] = float(np.clip(out[k], lo, hi))
        return out

    # ------------------------------------------------------------------ Suction + routing
    def _set_suction_adhesion_ctrl(self, ctrl: float) -> None:
        self.data.ctrl[self.suction_adhesion_actuator_id] = float(np.clip(ctrl, 0.0, 1.0))

    def _free_end_contact_metrics(self) -> Dict[str, float | bool]:
        free_pos = self._free_end_pos()
        cup_pos = self._suction_pos()
        delta = free_pos - cup_pos
        dist = float(np.linalg.norm(delta))

        if dist > 1e-9:
            free_dir = delta / dist
        else:
            free_dir = np.array([0.0, 0.0, 1.0], dtype=float)

        cup_axis = self.data.site_xmat[self.suction_site].reshape(3, 3)[:, 2]
        alignment_cos = float(abs(np.dot(cup_axis, free_dir)))

        in_radius = bool(dist <= self.suction_radius)
        alignment_ok = bool(alignment_cos >= self.suction_alignment_cos_min)
        attach_eligible = bool(in_radius and alignment_ok)

        dist_score = float(np.clip(1.0 - dist / self.suction_radius, 0.0, 1.0))
        align_scale = max(1.0 - self.suction_alignment_cos_min, 1e-6)
        align_score = float(
            np.clip((alignment_cos - self.suction_alignment_cos_min) / align_scale, 0.0, 1.0)
        )
        quality = float(np.clip(0.7 * dist_score + 0.3 * align_score, 0.0, 1.0))

        if attach_eligible:
            if self.suction_attach_stochastic:
                attach_probability = float(
                    np.clip(
                        self.suction_success_prob_floor
                        + (self.suction_success_prob_ceiling - self.suction_success_prob_floor)
                        * quality,
                        0.0,
                        1.0,
                    )
                )
            else:
                attach_probability = 1.0
        else:
            attach_probability = 0.0

        return {
            "cup_free_dist": dist,
            "suction_alignment_cos": alignment_cos,
            "suction_attach_quality": quality,
            "suction_attach_prob": attach_probability,
            "suction_attach_eligible": attach_eligible,
            "suction_distance_ok": in_radius,
            "suction_alignment_ok": alignment_ok,
        }

    def _routing_metrics(self) -> Dict[str, float | int | bool]:
        target_idx = min(self.routed_clip_count, self.n_clips - 1)
        target = self.data.site_xpos[self._clip_site_ids[target_idx]].copy()
        free_pos = self._free_end_pos()
        delta = free_pos - target

        planar_dist = float(np.linalg.norm(delta[:2]))
        vertical_dist = float(abs(delta[2]))
        in_capture = bool(
            planar_dist <= self.clip_capture_radius and vertical_dist <= self.clip_capture_z_tol
        )

        return {
            "target_clip_index": int(target_idx),
            "target_planar_dist": planar_dist,
            "target_vertical_dist": vertical_dist,
            "target_in_capture": in_capture,
        }

    def _update_suction(self, suction_cmd: float) -> float:
        reward = 0.0
        self._last_attach_attempted = False
        self._last_attach_success = False
        self._last_attach_eligible = False
        self._last_attach_prob = 0.0
        self._last_attach_quality = 0.0

        suction_cmd = float(max(0.0, suction_cmd))
        if suction_cmd <= 1e-6:
            self.suction_on = False
            self.grasped = False
            self._set_suction_adhesion_ctrl(0.0)
            return reward

        self.suction_on = True
        metrics = self._free_end_contact_metrics()

        if self.grasped:
            if float(metrics["cup_free_dist"]) > (self.suction_radius * self.suction_detach_dist_scale):
                self.grasped = False

        if not self.grasped:
            self._last_attach_attempted = True
            self._last_attach_eligible = bool(metrics["suction_attach_eligible"])
            self._last_attach_prob = float(metrics["suction_attach_prob"])
            self._last_attach_quality = float(metrics["suction_attach_quality"])

            if self._last_attach_eligible and self._rng.uniform() <= self._last_attach_prob:
                self.grasped = True
                self._last_attach_success = True
                reward += 0.20
            elif (
                self.suction_attach_fail_penalty > 0.0
                and bool(metrics["suction_distance_ok"])
                and self.suction_on
            ):
                reward -= self.suction_attach_fail_penalty

        self._set_suction_adhesion_ctrl(suction_cmd if self.grasped else 0.0)
        return reward

    def _update_clip_progress(self) -> float:
        self._last_progress_advanced = False
        if self.routed_clip_count >= self.n_clips:
            return 0.0

        routing = self._routing_metrics()
        in_capture = bool(routing["target_in_capture"])
        if in_capture:
            self._clip_hold_steps += 1
        else:
            self._clip_hold_steps = 0

        reward = 0.0
        planar_dist = float(routing["target_planar_dist"])
        reward += 0.35 * float(self._prev_target_planar_dist - planar_dist)
        self._prev_target_planar_dist = planar_dist

        if self._clip_hold_steps >= self.clip_hold_steps_required:
            self.routed_clip_count += 1
            self._clip_hold_steps = 0
            self._last_progress_advanced = True
            reward += self.clip_progress_bonus
            if self.routed_clip_count < self.n_clips:
                self._prev_target_planar_dist = self._routing_metrics()["target_planar_dist"]

        return reward

    # ------------------------------------------------------------------ Reward / obs
    def _compute_reward(self) -> Tuple[float, bool, Dict]:
        routing = self._routing_metrics()
        suction_metrics = self._free_end_contact_metrics()

        free_pos = self._free_end_pos()
        free_vel = self._free_end_vel()
        anchor_pos = self._anchor_pos()

        target_dist = float(routing["target_planar_dist"])
        target_z_dist = float(routing["target_vertical_dist"])
        progress_fraction = float(self.routed_clip_count / self.n_clips)

        reward = -0.002
        reward -= 0.75 * target_dist
        reward -= 0.08 * target_z_dist
        reward += 0.20 * progress_fraction
        if self.suction_on:
            reward += 0.01
        if self.grasped:
            reward += 0.08

        completion_stable = bool(np.linalg.norm(free_vel) < 0.30)
        success = bool(self.routed_clip_count >= self.n_clips and completion_stable)
        if success:
            reward += self.completion_bonus

        time_out = bool(self.time_limit > 0 and self.step_id >= self.time_limit)
        done = bool(success or time_out)

        info = {
            "success": success,
            "time_out": time_out,
            "grasped": bool(self.grasped),
            "suction_on": bool(self.suction_on),
            "suction_ctrl": float(self.data.ctrl[self.suction_adhesion_actuator_id]),
            "routed_clip_count": int(self.routed_clip_count),
            "num_clips": int(self.n_clips),
            "route_progress_fraction": progress_fraction,
            "target_clip_index": int(routing["target_clip_index"]),
            "target_clip_planar_dist": target_dist,
            "target_clip_vertical_dist": target_z_dist,
            "target_clip_in_capture": bool(routing["target_in_capture"]),
            "target_clip_hold_steps": int(self._clip_hold_steps),
            "target_clip_hold_required": int(self.clip_hold_steps_required),
            "progressed_clip_this_step": bool(self._last_progress_advanced),
            "free_end_height": float(free_pos[2]),
            "free_end_speed": float(np.linalg.norm(free_vel)),
            "anchor_to_free_dist": float(np.linalg.norm(free_pos - anchor_pos)),
            "suction_last_attach_attempted": bool(self._last_attach_attempted),
            "suction_last_attach_success": bool(self._last_attach_success),
            "suction_last_attach_eligible": bool(self._last_attach_eligible),
            "suction_last_attach_prob": float(self._last_attach_prob),
            "suction_last_attach_quality": float(self._last_attach_quality),
        }
        info.update(suction_metrics)
        return float(reward), done, info

    def _observe(self) -> np.ndarray:
        qpos = self.data.qpos
        qvel = self.data.qvel

        joint_pos = np.array([qpos[qid] for qid in self._robot_qpos_ids], dtype=np.float32)
        joint_vel = np.array([qvel[did] for did in self.robot_dofs], dtype=np.float32)
        suction_pos = self._suction_pos().astype(np.float32)
        free_pos = self._free_end_pos().astype(np.float32)
        free_vel = self._free_end_vel().astype(np.float32)

        target_idx = min(self.routed_clip_count, self.n_clips - 1)
        current_clip_pos = self.data.site_xpos[self._clip_site_ids[target_idx]].astype(np.float32)
        all_clip_pos = np.array(
            [self.data.site_xpos[sid] for sid in self._clip_site_ids],
            dtype=np.float32,
        ).reshape(-1)

        free_to_target = (current_clip_pos - free_pos).astype(np.float32)
        anchor_to_free = (free_pos - self._anchor_pos().astype(np.float32)).astype(np.float32)

        obs = np.concatenate(
            [
                joint_pos,
                joint_vel,
                suction_pos,
                free_pos,
                free_vel,
                current_clip_pos,
                all_clip_pos,
                free_to_target,
                anchor_to_free,
                np.array([self.routed_clip_count / self.n_clips], dtype=np.float32),
                np.array([1.0 if self.suction_on else 0.0], dtype=np.float32),
                np.array([1.0 if self.grasped else 0.0], dtype=np.float32),
                self._last_action.astype(np.float32),
            ]
        ).astype(np.float32)
        return obs

    # ------------------------------------------------------------------ Step / render
    def step(self, action: Iterable[float]) -> StepResult:
        act = np.asarray(action, dtype=float).flatten()
        if act.shape[0] != self.action_dim:
            raise ValueError(f"action should have shape ({self.action_dim},)")
        act = np.clip(act, -1.0, 1.0)
        self._last_action = act.astype(np.float32)

        delta_pos = act[:3] * self.ee_step
        dyaw = float(act[3] * self.yaw_step)
        suction_cmd = float(np.clip(act[4], 0.0, 1.0))

        target_pos, target_yaw = self._desired_ee(delta_pos, dyaw)
        qvel_cmd = self._ik_cartesian(target_pos, target_yaw)
        qvel_cmd = np.clip(qvel_cmd, -self.max_joint_vel, self.max_joint_vel)

        if np.linalg.norm(act[:4]) < self.hold_eps:
            qpos_targets = self._last_targets.copy()
        else:
            qpos_targets = self._last_targets + qvel_cmd * self.ik_gain
        qpos_targets = self._clamp_to_limits(qpos_targets)
        self._last_targets = qpos_targets.copy()

        for k, act_id in enumerate(self.robot_actuators):
            self.data.ctrl[act_id] = qpos_targets[k]

        suction_reward = self._update_suction(suction_cmd)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        progress_reward = self._update_clip_progress()
        shaped, done, info = self._compute_reward()

        reward = float(suction_reward + progress_reward + shaped)
        self.step_id += 1
        self._last_reward = reward
        self._last_step_info = dict(info)
        return StepResult(self._observe(), reward, done, info)

    def _compose_multi_camera_frame(self) -> np.ndarray:
        self._renderer_overview.update_scene(self.data, camera=self._camera_overview)
        frame_overview = self._renderer_overview.render()
        self._renderer_board.update_scene(self.data, camera=self._camera_board)
        frame_board = self._renderer_board.render()
        self._renderer_side.update_scene(self.data, camera=self._camera_side)
        frame_side = self._renderer_side.render()
        self._renderer_ee.update_scene(self.data, camera=self._camera_ee)
        frame_ee = self._renderer_ee.render()

        top_row = np.concatenate([frame_overview, frame_board], axis=1)
        bottom_row = np.concatenate([frame_side, frame_ee], axis=1)
        return np.concatenate([top_row, bottom_row], axis=0)

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if mode == "human":
            return None
        if mode == "rgb_array":
            return self._compose_multi_camera_frame()
        raise ValueError("mode must be 'human' or 'rgb_array'")

    def close(self) -> None:
        for renderer_name in (
            "_renderer_overview",
            "_renderer_board",
            "_renderer_side",
            "_renderer_ee",
        ):
            renderer = getattr(self, renderer_name, None)
            if renderer is None:
                continue
            if hasattr(renderer, "close"):
                renderer.close()
            elif hasattr(renderer, "free"):
                renderer.free()
            setattr(self, renderer_name, None)

    def sample_action(self) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)
