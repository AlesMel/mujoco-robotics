"""URSlotSorterEnv — pick-and-place with color-matched slots.

The task: control a 6-DOF UR arm to pick colored objects and drop
them into matching slots inside a tray with dividers.

RL usage (Gymnasium)::

    from mujoco_robot.envs import SlotSorterGymnasium
    env = SlotSorterGymnasium()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Teleop::

    from mujoco_robot.envs import URSlotSorterEnv
    from mujoco_robot.teleop import SlotSorterTeleop
    env = URSlotSorterEnv(time_limit=0)
    SlotSorterTeleop(env).run()
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

from mujoco_robot.core.xml_builder import load_robot_xml


# ---------------------------------------------------------------------------
# Default model path — uses the packaged ur5e.xml
# ---------------------------------------------------------------------------
_DEFAULT_MODEL = str(
    Path(__file__).resolve().parent.parent / "robots" / "ur5e.xml"
)

# Fallback robot XML when the file is missing (old-style, no dual-geom)
_FALLBACK_ROBOT_XML: str = ""
try:
    _FALLBACK_ROBOT_XML = Path(_DEFAULT_MODEL).read_text()
except FileNotFoundError:
    pass


@dataclass
class StepResult:
    """Container returned by :meth:`URSlotSorterEnv.step`."""
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict


# ---------------------------------------------------------------------------
# Gymnasium wrapper
# ---------------------------------------------------------------------------
class SlotSorterGymnasium(gymnasium.Env):
    """Gymnasium wrapper around :class:`URSlotSorterEnv` for SB3/VecEnv use.

    Parameters
    ----------
    seed : int | None
        Random seed.
    render : bool
        If ``True``, use high-resolution rendering (for video capture).
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, seed: int | None = None, render: bool = False):
        self.base = URSlotSorterEnv(
            render_size=(160, 120) if not render else (640, 480),
            time_limit=400,
            seed=seed,
        )
        self.action_space = gymnasium.spaces.Box(
            -1.0, 1.0, shape=(self.base.action_dim,), dtype=np.float32
        )
        self.observation_space = gymnasium.spaces.Box(
            -np.inf, np.inf, shape=(self.base.observation_dim,), dtype=np.float32
        )
        self.render_mode = "rgb_array" if render else None

    def reset(self, *, seed: int | None = None, options=None) -> Tuple[np.ndarray, dict]:
        obs = self.base.reset(seed=seed)
        return obs.astype(np.float32), {}

    def step(self, action):
        res: StepResult = self.base.step(action)
        success_complete = res.info.get("successes", 0) >= self.base.n_objects
        time_limit_hit = (
            self.base.time_limit > 0
            and self.base.step_id >= self.base.time_limit
        )
        terminated = success_complete
        truncated = time_limit_hit and not success_complete
        return res.obs, res.reward, terminated, truncated, res.info

    def render(self):
        return (
            self.base.render(mode="rgb_array")
            if self.render_mode == "rgb_array"
            else None
        )

    def close(self):
        self.base.close()


# ---------------------------------------------------------------------------
# Core environment
# ---------------------------------------------------------------------------
class URSlotSorterEnv:
    """UR-style pick-and-place with compartmentalised slots.

    Action space (5-D, continuous [-1, 1]):
        ``[dx, dy, dz, dyaw, grip]`` — Cartesian EE velocity + gripper.

    Parameters
    ----------
    model_path : str
        Path to the robot MJCF XML.
    min_objects / max_objects : int
        Object count range.
    time_limit : int
        Max steps per episode (0 = unlimited).
    ee_step : float
        EE position step size per action unit.
    yaw_step : float
        EE yaw step size per action unit.
    grasp_radius : float
        Distance at which the magnetic grasp activates.
    render_size : tuple[int, int]
        Width × height of the offscreen renderer.
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        min_objects: int = 5,
        max_objects: int = 5,
        time_limit: int = 400,
        table_half: float = 0.55,
        slot_half: float = 0.35,
        slot_radius: float = 0.05,
        ee_step: float = 0.06,
        yaw_step: float = 0.6,
        grasp_radius: float = 0.06,
        render_size: Tuple[int, int] = (960, 720),
        seed: Optional[int] = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)

        self.model_path = model_path
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.table_half = table_half
        self.slot_half = slot_half
        self.slot_radius = slot_radius
        self.ee_step = ee_step
        self.yaw_step = yaw_step
        self.grasp_radius = grasp_radius
        self.time_limit = time_limit
        self.render_size = render_size

        # Stability tuning
        self.max_joint_vel = 4.0
        self.ik_damping = 0.02
        self.ee_obj_shaping = 0.8
        self.init_q = np.array([-np.pi, -np.pi / 2, np.pi / 2,
                                -np.pi / 2, -np.pi / 2, 0.0])
        self.settle_steps = 500
        self.hold_eps = 0.05
        self.n_substeps = 5
        self.spawn_xy_bounds = (-0.05, 0.18, -0.20, 0.20)

        # Visual settings
        self.colors = [
            np.array([0.85, 0.2, 0.2, 1.0]),
            np.array([0.1, 0.7, 0.2, 1.0]),
            np.array([0.2, 0.2, 0.9, 1.0]),
            np.array([0.9, 0.7, 0.2, 1.0]),
            np.array([0.6, 0.2, 0.8, 1.0]),
        ]
        self.shape_types = ["box", "cylinder"]
        self.obj_box_half = np.array([0.035, 0.035, 0.035], dtype=float)
        self.obj_cyl_radius = 0.03
        self.obj_cyl_halfheight = 0.03
        self.slot_clearance = 0.01
        self.slot_shapes = [
            self.shape_types[i % len(self.shape_types)]
            for i in range(self.max_objects)
        ]
        self.obj_half_heights = np.array(
            [self.obj_box_half[2] if s == "box" else self.obj_cyl_halfheight
             for s in self.slot_shapes],
            dtype=float,
        )
        self.slot_box_accept = np.full((self.max_objects, 2), np.nan, dtype=float)
        self.slot_cyl_accept = np.full((self.max_objects,), np.nan, dtype=float)
        self.slot_proximity_radius = np.full((self.max_objects,), np.nan, dtype=float)
        self.tray_surface_z = 0.0

        # ---- Build MuJoCo model ----
        robot_xml = self._load_robot_xml(model_path)
        self.model_xml = self._build_env_xml(robot_xml)
        self.model = mujoco.MjModel.from_xml_string(self.model_xml)
        self.data = mujoco.MjData(self.model)

        # Solver settings
        self.model.opt.timestep = 0.002
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.iterations = max(self.model.opt.iterations, 50)
        self.model.opt.noslip_iterations = max(self.model.opt.noslip_iterations, 10)
        self.model.opt.gravity[:] = np.array([0.0, 0.0, -9.81])

        self.renderer = mujoco.Renderer(
            self.model, height=render_size[1], width=render_size[0]
        )

        # Look up IDs
        self.ee_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
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
        self._qvel_lp = np.zeros(len(self.robot_joints), dtype=float)
        self._last_targets = self.init_q.copy()

        # Object / slot IDs
        self.object_bodies = [f"obj{i}" for i in range(self.max_objects)]
        self.object_geoms = [f"obj{i}_geom" for i in range(self.max_objects)]
        self.object_freejoints = [f"obj{i}_free" for i in range(self.max_objects)]
        self.slot_sites = [f"slot{i}_site" for i in range(self.max_objects)]
        self.slot_markers = [f"slot{i}_marker" for i in range(self.max_objects)]

        self.obj_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in self.object_bodies
        ]
        self.obj_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in self.object_geoms
        ]
        self.obj_free_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self.object_freejoints
        ]
        self.slot_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
            for n in self.slot_sites
        ]
        self.slot_marker_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in self.slot_markers
        ]

        # Passive damping / friction
        for dof in self.robot_dofs:
            self.model.dof_damping[dof] = max(self.model.dof_damping[dof], 8.0)
            self.model.dof_frictionloss[dof] = max(self.model.dof_frictionloss[dof], 0.5)
        for k, act in enumerate(self.robot_actuators):
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, self.robot_joints[k]
            )
            rng = self.model.jnt_range[jid]
            low, high = (-np.pi, np.pi) if rng[0] >= rng[1] else rng
            self.model.actuator_ctrlrange[act] = np.array([low, high], dtype=float)
            if self.model.actuator_gainprm[act, 0] <= 0.0:
                self.model.actuator_gainprm[act, 0] = 400.0

        # Store defaults for domain randomization
        self.obj_base_mass = self.model.body_mass[self.obj_body_ids].copy()
        self.obj_base_friction = self.model.geom_friction[self.obj_geom_ids].copy()

        # State
        self.step_id = 0
        self.n_objects = self.max_objects
        self.grasped = -1
        self.gripper_closed = False
        self.success_mask = np.zeros(self.max_objects, dtype=bool)
        self.grasp_offsets: List[np.ndarray] = [
            np.zeros(3) for _ in range(self.max_objects)
        ]
        self.placed_once = np.zeros(self.max_objects, dtype=bool)

    # ------------------------------------------------------------------ XML
    def _load_robot_xml(self, path: str) -> str:
        from mujoco_robot.core.xml_builder import load_robot_xml
        try:
            return load_robot_xml(path)
        except FileNotFoundError:
            if _FALLBACK_ROBOT_XML:
                return _FALLBACK_ROBOT_XML
            raise FileNotFoundError(f"Robot MJCF not found at '{path}'.")

    def _build_env_xml(self, robot_xml: str) -> str:
        """Inject tray, slots, and objects into the robot MJCF."""
        root = ET.fromstring(robot_xml)

        # Framebuffer
        w, h = self.render_size
        visual = root.find("visual")
        if visual is None:
            visual = ET.SubElement(root, "visual")
        global_elem = visual.find("global")
        if global_elem is None:
            global_elem = ET.SubElement(visual, "global")
        global_elem.set("offwidth", str(max(w, 640)))
        global_elem.set("offheight", str(max(h, 480)))

        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("Robot MJCF missing <worldbody>")

        # Disable self-collisions on arm links; keep EE contact
        for geom in root.findall(".//geom"):
            name = geom.get("name", "")
            if name.startswith("link") or name == "base_geom":
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
            if name == "ee_sphere":
                geom.set("contype", "1")
                geom.set("conaffinity", "1")
                geom.set("friction", "1 0.05 0.01")

        # Contact exclusion pairs
        contact_elem = root.find("contact")
        if contact_elem is None:
            contact_elem = ET.SubElement(root, "contact")
        robot_body_pairs = [
            ("base", "shoulder"), ("shoulder", "upper_arm"),
            ("upper_arm", "forearm"), ("forearm", "wrist1"),
            ("wrist1", "wrist2"), ("wrist2", "wrist3"),
            ("base", "upper_arm"), ("base", "forearm"),
            ("shoulder", "forearm"), ("shoulder", "wrist1"),
            ("upper_arm", "wrist1"), ("upper_arm", "wrist2"),
            ("forearm", "wrist2"), ("forearm", "wrist3"),
        ]
        for b1, b2 in robot_body_pairs:
            ET.SubElement(contact_elem, "exclude", {"body1": b1, "body2": b2})

        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")
        option.set("o_margin", "0.01")

        # ---------- Tray ----------
        tray_x, tray_y, tray_z = 0.28, 0.0, 0.73
        tray_half = np.array([0.18, 0.28, 0.01])
        wall_t, wall_h = 0.01, 0.07
        compartment_half_x = 0.055
        compartment_half_y = 0.042

        self.tray_surface_z = tray_z + float(tray_half[2])

        tray = ET.Element("body", {"name": "tray", "pos": f"{tray_x} {tray_y} {tray_z}"})
        ET.SubElement(tray, "geom", {
            "name": "tray_base", "type": "box", "pos": "0 0 0",
            "size": f"{tray_half[0]} {tray_half[1]} {tray_half[2]}",
            "rgba": "0.82 0.82 0.88 1", "friction": "1 0.1 0.05",
            "contype": "1", "conaffinity": "1",
        })

        wall_z = tray_half[2] + wall_h / 2
        walls = [
            ("tray_wall_n", f"0 {tray_half[1]-wall_t/2} {wall_z}",
             f"{tray_half[0]} {wall_t/2} {wall_h/2}"),
            ("tray_wall_s", f"0 {-tray_half[1]+wall_t/2} {wall_z}",
             f"{tray_half[0]} {wall_t/2} {wall_h/2}"),
            ("tray_wall_e", f"{tray_half[0]-wall_t/2} 0 {wall_z}",
             f"{wall_t/2} {tray_half[1]} {wall_h/2}"),
            ("tray_wall_w", f"{-tray_half[0]+wall_t/2} 0 {wall_z}",
             f"{wall_t/2} {tray_half[1]} {wall_h/2}"),
        ]
        for name, pos, size in walls:
            ET.SubElement(tray, "geom", {
                "name": name, "type": "box", "pos": pos, "size": size,
                "rgba": "0.7 0.7 0.75 1", "contype": "1", "conaffinity": "1",
            })

        y_min = -tray_half[1] + compartment_half_y + wall_t * 2
        y_max = tray_half[1] - compartment_half_y - wall_t * 2
        centers_y = np.linspace(y_min, y_max, self.max_objects)
        self.slot_centers = []
        self.slot_half_extents = np.array(
            [compartment_half_x, compartment_half_y, tray_half[2]], dtype=float
        )

        # Dividers
        for i in range(self.max_objects - 1):
            y_div = (centers_y[i] + centers_y[i + 1]) / 2
            ET.SubElement(tray, "geom", {
                "name": f"tray_divider_{i}", "type": "box",
                "pos": f"0 {y_div} {wall_z}",
                "size": f"{tray_half[0]} {wall_t/2} {wall_h/2}",
                "rgba": "0.65 0.65 0.7 1", "contype": "1", "conaffinity": "1",
            })

        # Slots
        for i in range(self.max_objects):
            cy = centers_y[i]
            shape = self.slot_shapes[i]
            color = self.colors[i % len(self.colors)]
            inset = wall_t * 1.5

            if shape == "box":
                hx = float(min(compartment_half_x - inset,
                               self.obj_box_half[0] + self.slot_clearance))
                hy = float(min(compartment_half_y - inset,
                               self.obj_box_half[1] + self.slot_clearance))
                self.slot_box_accept[i] = np.array([hx, hy], dtype=float)
                self.slot_proximity_radius[i] = max(hx, hy)
                slot_center_z = self.tray_surface_z + float(self.obj_box_half[2])
                outline_type, outline_size = "box", f"{hx} {hy} 0.001"
                site_type, site_size = "box", f"{hx} {hy} 0.001"
            else:
                r = float(min(
                    min(compartment_half_x, compartment_half_y) - inset,
                    self.obj_cyl_radius + self.slot_clearance,
                ))
                self.slot_cyl_accept[i] = r
                self.slot_proximity_radius[i] = r
                slot_center_z = self.tray_surface_z + float(self.obj_cyl_halfheight)
                outline_type, outline_size = "cylinder", f"{r} 0.001"
                site_type, site_size = "cylinder", f"{r} 0.001"

            self.slot_centers.append(
                np.array([tray_x, tray_y + cy, slot_center_z], dtype=float)
            )

            # Compartment lips
            ET.SubElement(tray, "geom", {
                "name": f"slot{i}_frontlip", "type": "box",
                "pos": f"{0} {cy - compartment_half_y - wall_t/2} {wall_z}",
                "size": f"{compartment_half_x} {wall_t/2} {wall_h/2}",
                "rgba": "0.68 0.68 0.72 1", "contype": "1", "conaffinity": "1",
            })
            ET.SubElement(tray, "geom", {
                "name": f"slot{i}_backlip", "type": "box",
                "pos": f"{0} {cy + compartment_half_y + wall_t/2} {wall_z}",
                "size": f"{compartment_half_x} {wall_t/2} {wall_h/2}",
                "rgba": "0.68 0.68 0.72 1", "contype": "1", "conaffinity": "1",
            })

            # Outline + marker + site
            ET.SubElement(tray, "geom", {
                "name": f"slot{i}_outline", "type": outline_type,
                "pos": f"{0} {cy} {tray_half[2] + 0.001}",
                "size": outline_size,
                "rgba": f"{color[0]} {color[1]} {color[2]} 0.18",
                "contype": "0", "conaffinity": "0",
            })
            ET.SubElement(tray, "geom", {
                "name": f"slot{i}_marker", "type": "box",
                "pos": f"{compartment_half_x*0.75} {cy} {tray_half[2] + 0.003}",
                "size": "0.01 0.018 0.0015",
                "rgba": f"{color[0]} {color[1]} {color[2]} 0.9",
                "contype": "0", "conaffinity": "0",
            })
            ET.SubElement(tray, "site", {
                "name": f"slot{i}_site",
                "pos": f"{0} {cy} {tray_half[2] + 0.001}",
                "size": site_size,
                "rgba": f"{color[0]} {color[1]} {color[2]} 0.06",
                "type": site_type,
            })

        worldbody.append(tray)

        # ---------- Objects ----------
        for i in range(self.max_objects):
            color = self.colors[i % len(self.colors)]
            shape = self.shape_types[i % len(self.shape_types)]
            obj_body = ET.Element("body", {"name": f"obj{i}", "pos": "0 0 0.9"})
            ET.SubElement(obj_body, "freejoint", {"name": f"obj{i}_free"})
            sz = (
                f"{self.obj_box_half[0]} {self.obj_box_half[1]} {self.obj_box_half[2]}"
                if shape == "box"
                else f"{self.obj_cyl_radius} {self.obj_cyl_halfheight}"
            )
            ET.SubElement(obj_body, "geom", {
                "name": f"obj{i}_geom", "type": shape, "size": sz,
                "rgba": " ".join(map(str, color)),
                "friction": "1.1 0.1 0.05", "mass": "0.08",
            })
            worldbody.append(obj_body)

        return ET.tostring(root, encoding="unicode")

    # ------------------------------------------------------------------ API
    @property
    def action_dim(self) -> int:
        """Action dimensionality: ``[dx, dy, dz, dyaw, grip]``."""
        return 5

    @property
    def observation_dim(self) -> int:
        """Observation dimensionality."""
        obj_block = self.max_objects * (6 + self.max_objects)
        return 6 + 6 + 4 + obj_block + self.max_objects * 3 + (self.max_objects + 1)

    # ------------------------------------------------------------------ Reset
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment and return the initial observation."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_id = 0
        self.grasped = -1
        self.gripper_closed = False
        self.success_mask[:] = False
        self.placed_once[:] = False
        self._qvel_lp[:] = 0
        self._last_targets = self.init_q.copy()
        self.n_objects = self.max_objects

        self._randomize_slots()
        self._randomize_objects()

        for qi, j in enumerate(self.robot_joints):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            adr = self.model.jnt_qposadr[jid]
            self.data.qpos[adr] = self.init_q[qi]
            vad = self.model.jnt_dofadr[jid]
            self.data.qvel[vad] = 0
            act_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_motor"
            )
            if act_id != -1:
                self.data.ctrl[act_id] = self.init_q[qi]

        self._move_to_spawn_home()

        for j in self.robot_joints:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            vad = self.model.jnt_dofadr[jid]
            self.data.qvel[vad] = 0
        mujoco.mj_forward(self.model, self.data)

        # Anchor IK yaw target to the home heading so pure-translation
        # commands don't cause unwanted yaw drift.
        self._target_yaw = self._ee_yaw()

        for _ in range(self.settle_steps):
            mujoco.mj_step(self.model, self.data)
        return self._observe()

    # ------------------------------------------------------------------ Randomization
    def _randomize_slots(self) -> None:
        self.slot_positions = np.array(self.slot_centers)
        for i, gid in enumerate(self.slot_marker_geom_ids):
            color = self.colors[i % len(self.colors)]
            rgba = self.model.geom_rgba[gid].copy()
            rgba[:3] = color[:3]
            rgba[3] = 0.25
            self.model.geom_rgba[gid] = rgba

    def _randomize_objects(self) -> None:
        z = 0.78
        xmin, xmax, ymin, ymax = self.spawn_xy_bounds
        min_dist = 0.07
        positions: List[np.ndarray] = []

        for i in range(self.n_objects):
            placed = False
            for _ in range(200):
                x = self._rng.uniform(xmin + 0.03, xmax - 0.03)
                y = self._rng.uniform(ymin + 0.03, ymax - 0.03)
                cand = np.array([x, y, z])
                if all(np.linalg.norm(cand[:2] - p[:2]) > min_dist for p in positions):
                    positions.append(cand)
                    placed = True
                    break
            if not placed:
                positions.append(np.array([xmin + 0.05 * (i + 1), ymin + 0.05, z]))

        for i in range(self.max_objects):
            body_id = self.obj_body_ids[i]
            geom_id = self.obj_geom_ids[i]
            qadr = self.model.jnt_qposadr[self.obj_free_ids[i]]
            vad = self.model.jnt_dofadr[self.obj_free_ids[i]]
            self.data.qvel[vad: vad + 6] = 0

            if i < self.n_objects:
                self.data.qpos[qadr: qadr + 3] = positions[i]
                rand_yaw = self._rng.uniform(-math.pi, math.pi)
                quat = np.zeros(4, dtype=float)
                mujoco.mju_axisAngle2Quat(
                    quat, np.array([0.0, 0.0, 1.0], dtype=float), rand_yaw
                )
                self.data.qpos[qadr + 3: qadr + 7] = quat
                mass_scale = self._rng.uniform(0.9, 1.1)
                fric_scale = self._rng.uniform(0.9, 1.1, size=3)
                self.model.body_mass[body_id] = self.obj_base_mass[i] * mass_scale
                self.model.geom_friction[geom_id] = self.obj_base_friction[i] * fric_scale
            else:
                self.data.qpos[qadr: qadr + 7] = np.array([3, 3, 3, 1, 0, 0, 0])
                self.model.body_mass[body_id] = self.obj_base_mass[i]
                self.model.geom_friction[geom_id] = self.obj_base_friction[i]
            self.grasp_offsets[i] = np.zeros(3)

    # ------------------------------------------------------------------ IK + control
    def _ee_yaw(self) -> float:
        mat = self.data.site_xmat[self.ee_site].reshape(3, 3)
        return math.atan2(mat[1, 0], mat[0, 0])

    def _desired_ee(
        self, delta_xyz: np.ndarray, dyaw: float
    ) -> Tuple[np.ndarray, float]:
        pos = self.data.site_xpos[self.ee_site].copy() + delta_xyz
        pos[0] = float(np.clip(pos[0], -self.table_half + 0.05, self.table_half - 0.05))
        pos[1] = float(np.clip(pos[1], -self.table_half + 0.05, self.table_half - 0.05))
        pos[2] = float(np.clip(pos[2], 0.76, 1.30))
        # Update persistent yaw target — pure translation (dyaw=0) keeps
        # the heading anchored instead of following kinematic drift.
        self._target_yaw = self._target_yaw + dyaw
        return pos, self._target_yaw

    def _ik_cartesian(
        self, target_pos: np.ndarray, target_yaw: float
    ) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site)

        pos_err = target_pos - self.data.site_xpos[self.ee_site]
        yaw_err = target_yaw - self._ee_yaw()
        yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
        target = np.concatenate([pos_err, np.array([yaw_err])])

        cols = self.robot_dofs
        J = np.vstack([jacp[:, cols], jacr[2:3, cols]])
        lam = float(self.ik_damping)
        JJ = J @ J.T + (lam * lam) * np.eye(J.shape[0])
        return J.T @ np.linalg.solve(JJ, target)

    def _move_to_spawn_home(self) -> None:
        """Drive the arm to a neutral pose above the spawn zone."""
        xmin, xmax, ymin, ymax = self.spawn_xy_bounds
        target_pos = np.array(
            [(xmin + xmax) * 0.5, (ymin + ymax) * 0.5, 1.0], dtype=float
        )
        target_yaw = 0.0
        dt = float(self.model.opt.timestep) * self.n_substeps
        self._last_targets = self.init_q.copy()

        for _ in range(180):
            qvel_cmd = self._ik_cartesian(target_pos, target_yaw)
            qvel_cmd = np.clip(qvel_cmd, -self.max_joint_vel, self.max_joint_vel)
            qpos_targets = self._clamp_to_limits(self._last_targets + qvel_cmd * dt)
            self._last_targets = qpos_targets.copy()
            for k, act_id in enumerate(self.robot_actuators):
                self.data.ctrl[act_id] = qpos_targets[k]
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0

    def _clamp_to_limits(self, targets: np.ndarray) -> np.ndarray:
        out = targets.copy()
        for k, j in enumerate(self.robot_joints):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            rng = self.model.jnt_range[jid]
            if rng[0] < rng[1]:
                out[k] = float(np.clip(out[k], rng[0], rng[1]))
        return out

    # ------------------------------------------------------------------ Grasp logic
    def _apply_grasp_weld(self) -> None:
        """Kinematically attach the grasped object to the end-effector."""
        if self.grasped == -1:
            return
        qadr = self.model.jnt_qposadr[self.obj_free_ids[self.grasped]]
        vad = self.model.jnt_dofadr[self.obj_free_ids[self.grasped]]
        target_pos = self.data.site_xpos[self.ee_site] + self.grasp_offsets[self.grasped]
        self.data.qpos[qadr: qadr + 3] = target_pos
        self.data.qvel[vad: vad + 6] = 0

    def _update_grasp(self, grip_pressed: bool) -> float:
        """Magnetic grasp: weld nearest object when gripper is closed."""
        reward = 0.0
        if grip_pressed and not self.gripper_closed:
            reward += 0.05
        self.gripper_closed = grip_pressed

        if not grip_pressed:
            self.grasped = -1
            return reward

        if self.grasped == -1:
            ee = self.data.site_xpos[self.ee_site]
            dists = []
            for i in range(self.n_objects):
                pos = self.data.xpos[self.obj_body_ids[i]]
                dists.append(np.linalg.norm(pos - ee))
            nearest = int(np.argmin(dists))
            if dists[nearest] < self.grasp_radius:
                self.grasped = nearest
                self.grasp_offsets[nearest] = (
                    self.data.xpos[self.obj_body_ids[nearest]] - ee
                )
                reward += 0.1

        self._apply_grasp_weld()
        return reward

    # ------------------------------------------------------------------ Reward
    def _distance_to_slot(self, obj_idx: int) -> float:
        obj_pos = self.data.xpos[self.obj_body_ids[obj_idx]]
        slot_pos = self.slot_positions[obj_idx]
        return float(np.linalg.norm(obj_pos - slot_pos))

    def _in_slot(self, obj_idx: int) -> bool:
        obj_pos = self.data.xpos[self.obj_body_ids[obj_idx]]
        slot = self.slot_positions[obj_idx]
        lin_vel = np.linalg.norm(self.data.cvel[self.obj_body_ids[obj_idx], 3:6])

        dx = float(obj_pos[0] - slot[0])
        dy = float(obj_pos[1] - slot[1])
        shape = self.slot_shapes[obj_idx]

        if shape == "box":
            hx, hy = self.slot_box_accept[obj_idx]
            xy_ok = (abs(dx) < hx * 0.9) and (abs(dy) < hy * 0.9)
        else:
            r = float(self.slot_cyl_accept[obj_idx])
            xy_ok = (dx * dx + dy * dy) < (r * 0.9) ** 2

        z_ok = abs(float(obj_pos[2] - slot[2])) < 0.02
        vel_ok = lin_vel < 0.05
        return bool(xy_ok and z_ok and vel_ok)

    def _compute_reward(self) -> Tuple[float, bool, Dict]:
        reward = -0.005
        done = False
        min_ee_obj = None

        if self.grasped == -1:
            ee = self.data.site_xpos[self.ee_site]
            dists = [
                np.linalg.norm(self.data.xpos[self.obj_body_ids[i]] - ee)
                for i in range(self.n_objects)
                if not self.success_mask[i]
            ]
            if dists:
                min_ee_obj = float(np.min(dists))
                reward -= self.ee_obj_shaping * min_ee_obj
                if self.gripper_closed and min_ee_obj < self.grasp_radius * 1.2:
                    reward += 0.1

        if self.grasped != -1:
            dist = self._distance_to_slot(self.grasped)
            reward -= 1.5 * dist

        newly_done = 0
        for i in range(self.n_objects):
            if self.success_mask[i]:
                continue
            if self._in_slot(i):
                self.success_mask[i] = True
                gid = self.slot_marker_geom_ids[i]
                base_rgba = self.model.geom_rgba[gid].copy()
                base_rgba[:3] = np.array([0.1, 1.0, 0.1])
                self.model.geom_rgba[gid] = base_rgba
                reward += 1.5
                newly_done += 1
                self.grasped = -1
        if newly_done > 0:
            reward += 0.2 * newly_done

        if self.success_mask[: self.n_objects].all():
            done = True
            reward += 2.0

        for i in range(self.n_objects):
            pos = self.data.xpos[self.obj_body_ids[i]]
            if pos[2] < 0.68:
                reward -= 0.3
            for j in range(self.n_objects):
                if j == i:
                    continue
                slot = self.slot_positions[j]
                if np.linalg.norm(pos[:2] - slot[:2]) < float(self.slot_proximity_radius[j]) * 0.9:
                    reward -= 0.05
                    break

        if self.time_limit > 0 and self.step_id >= self.time_limit:
            done = True

        info = {
            "successes": int(np.sum(self.success_mask)),
            "grasped": self.grasped,
            "n_objects": self.n_objects,
            "min_ee_obj_dist": min_ee_obj,
        }
        return float(reward), done, info

    # ------------------------------------------------------------------ Observation
    def _observe(self) -> np.ndarray:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        obs_parts: List[np.ndarray] = []

        for j in self.robot_joints:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            adr = self.model.jnt_qposadr[jid]
            obs_parts.append(np.array([qpos[adr]]))
        for j in self.robot_joints:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            adr = self.model.jnt_dofadr[jid]
            obs_parts.append(np.array([qvel[adr]]))

        obs_parts.append(self.data.site_xpos[self.ee_site].copy())
        obs_parts.append(np.array([self._ee_yaw()]))

        for i in range(self.max_objects):
            qadr = self.model.jnt_qposadr[self.obj_free_ids[i]]
            vad = self.model.jnt_dofadr[self.obj_free_ids[i]]
            pos = qpos[qadr: qadr + 3]
            vel = qvel[vad: vad + 3]
            obs_parts.append(pos if i < self.n_objects else np.zeros(3))
            obs_parts.append(vel if i < self.n_objects else np.zeros(3))
            onehot = np.zeros(self.max_objects, dtype=float)
            if i < self.n_objects:
                onehot[i] = 1.0
            obs_parts.append(onehot)

        for i in range(self.max_objects):
            pos = self.slot_positions[i] if i < self.n_objects else np.zeros(3)
            obs_parts.append(pos)

        grasp_hot = np.zeros(self.max_objects + 1, dtype=float)
        if 0 <= self.grasped < self.max_objects:
            grasp_hot[self.grasped] = 1.0
        else:
            grasp_hot[-1] = 1.0
        obs_parts.append(grasp_hot)

        return np.concatenate(obs_parts).astype(np.float32)

    # ------------------------------------------------------------------ Step / render
    def step(self, action: Iterable[float]) -> StepResult:
        """Execute one environment step.

        Parameters
        ----------
        action : array-like, shape (5,)
            ``[dx, dy, dz, dyaw, grip]`` each in [-1, 1].
        """
        act = np.asarray(action, dtype=float)
        if act.shape[0] != self.action_dim:
            raise ValueError(f"action should have shape ({self.action_dim},)")
        act = np.clip(act, -1.0, 1.0)

        delta_pos = act[:3] * self.ee_step
        dyaw = act[3] * self.yaw_step
        grip = act[4] > 0.0

        target_pos, target_yaw = self._desired_ee(delta_pos, dyaw)
        qvel_cmd = self._ik_cartesian(target_pos, target_yaw)
        qvel_cmd = np.clip(qvel_cmd, -self.max_joint_vel, self.max_joint_vel)

        ik_gain = 0.15
        if np.linalg.norm(act[:4]) < self.hold_eps:
            qpos_targets = self._last_targets.copy()
        else:
            qpos_targets = self._last_targets + qvel_cmd * ik_gain
        qpos_targets = self._clamp_to_limits(qpos_targets)
        self._last_targets = qpos_targets.copy()
        for k, act_id in enumerate(self.robot_actuators):
            self.data.ctrl[act_id] = qpos_targets[k]

        reward = self._update_grasp(grip)
        self._apply_grasp_weld()
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
            self._apply_grasp_weld()

        shaped, done, info = self._compute_reward()
        reward += shaped
        self.step_id += 1
        return StepResult(self._observe(), float(reward), done, info)

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the scene."""
        if mode == "human":
            return None
        if mode == "rgb_array":
            self.renderer.update_scene(self.data, camera="top")
            return self.renderer.render()
        raise ValueError("mode must be 'human' or 'rgb_array'")

    def close(self) -> None:
        """Release rendering resources."""
        if hasattr(self.renderer, "close"):
            self.renderer.close()
        elif hasattr(self.renderer, "free"):
            self.renderer.free()
        self.renderer = None

    def sample_action(self) -> np.ndarray:
        """Sample a random action."""
        return self._rng.uniform(-1, 1, size=self.action_dim).astype(np.float32)
