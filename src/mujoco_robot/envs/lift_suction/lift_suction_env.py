"""UR3e lift task with a Robotiq EPick-style suction end-effector.

This environment models a single-object lifting task:
- control UR3e in Cartesian delta-space + yaw
- toggle suction on/off
- pick one cube and lift it to a target height

The EPick body uses the real STL mesh from the ROS2 EPick description package,
with collision and suction-cup dimensions mirrored from that xacro.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import gymnasium
import mujoco
import numpy as np

from mujoco_robot.core.xml_builder import load_robot_xml
from mujoco_robot.robots.actuators import (
    configure_position_actuators,
    resolve_robot_actuators,
)


_DEFAULT_MODEL = str(
    Path(__file__).resolve().parent.parent.parent / "robots" / "ur3e.xml"
)
_EPICK_BODY_MESH = (
    Path(__file__).resolve().parent.parent.parent
    / "robots"
    / "assets"
    / "epick"
    / "epick_body.stl"
)
_EPICK_BODY_LENGTH = 0.1023
_EPICK_BODY_COLLISION_RADIUS = 0.044
_EPICK_CUP_RADIUS = 0.012
_EPICK_CUP_HEIGHT = 0.015

_FALLBACK_ROBOT_XML: str = ""
try:
    _FALLBACK_ROBOT_XML = Path(_DEFAULT_MODEL).read_text()
except FileNotFoundError:
    pass


@dataclass
class StepResult:
    """Container returned by :meth:`URLiftSuctionEnv.step`."""

    obs: np.ndarray
    reward: float
    done: bool
    info: Dict


class LiftSuctionGymnasium(gymnasium.Env):
    """Gymnasium wrapper around :class:`URLiftSuctionEnv`."""

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        seed: int | None = None,
        render: bool = False,
        time_limit: int = 300,
        model_path: str = _DEFAULT_MODEL,
        actuator_profile: str = "ur3e",
        **env_kwargs,
    ):
        self.base = URLiftSuctionEnv(
            model_path=model_path,
            actuator_profile=actuator_profile,
            render_size=(160, 120) if not render else (640, 480),
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
        self.render_mode = "rgb_array" if render else None

    def reset(self, *, seed: int | None = None, options=None):
        obs = self.base.reset(seed=seed)
        return obs.astype(np.float32), {}

    def step(self, action):
        res: StepResult = self.base.step(action)
        terminated = bool(res.info.get("success", False))
        truncated = bool(res.info.get("time_out", False) and not terminated)
        return res.obs, res.reward, terminated, truncated, res.info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.base.render(mode="rgb_array")
        return None

    def close(self):
        self.base.close()


class URLiftSuctionEnv:
    """UR3e single-object lift task with suction grasping."""

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        actuator_profile: str = "ur3e",
        time_limit: int = 300,
        ee_step: float = 0.05,
        yaw_step: float = 0.5,
        suction_radius: float = 0.03,
        lift_height: float = 0.18,
        render_size: Tuple[int, int] = (960, 720),
        seed: Optional[int] = None,
        settle_steps: int = 150,
    ) -> None:
        self._rng = np.random.default_rng(seed)

        self.model_path = model_path
        self.actuator_profile = actuator_profile
        self.time_limit = int(time_limit)
        self.ee_step = float(ee_step)
        self.yaw_step = float(yaw_step)
        self.suction_radius = float(suction_radius)
        self.lift_height = float(lift_height)
        self.render_size = render_size
        self.settle_steps = int(max(0, settle_steps))

        # Control / stability settings.
        self.max_joint_vel = 4.0
        self.ik_damping = 0.02
        self.ik_gain = 0.15
        self.hold_eps = 0.02
        self.n_substeps = 5
        self.init_q = np.array(
            [-math.pi, -math.pi / 2.0, math.pi / 2.0, -math.pi / 2.0, -math.pi / 2.0, 0.0],
            dtype=float,
        )

        self.spawn_xy_bounds = (0.04, 0.24, -0.18, 0.18)
        self.obj_half = np.array([0.02, 0.02, 0.02], dtype=float)

        robot_xml = self._load_robot_xml(model_path)
        self.model_xml = self._build_env_xml(robot_xml)
        self.model = mujoco.MjModel.from_xml_string(self.model_xml)
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = 0.002
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.iterations = max(self.model.opt.iterations, 50)
        self.model.opt.gravity[:] = np.array([0.0, 0.0, -9.81])

        self.renderer = mujoco.Renderer(
            self.model, height=render_size[1], width=render_size[0]
        )

        self.ee_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.suction_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "suction_site"
        )
        if self.suction_site < 0:
            # Fallback keeps env functional even if model has no custom site.
            self.suction_site = self.ee_site

        self.obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "lift_obj")
        self.obj_free_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "lift_obj_free"
        )
        self.goal_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "lift_goal"
        )

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

        self.step_id = 0
        self.suction_on = False
        self.grasped = False
        self._grasp_offset = np.zeros(3, dtype=float)
        self._target_yaw = 0.0
        self._last_targets = self.init_q.copy()
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._spawn_pos = np.zeros(3, dtype=float)
        self._goal_pos = np.zeros(3, dtype=float)

    # ------------------------------------------------------------------ XML
    def _load_robot_xml(self, path: str) -> str:
        try:
            return load_robot_xml(path)
        except FileNotFoundError:
            if _FALLBACK_ROBOT_XML:
                return _FALLBACK_ROBOT_XML
            raise FileNotFoundError(f"Robot MJCF not found at '{path}'.")

    def _build_env_xml(self, robot_xml: str) -> str:
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

        # EPick chain (dimensions from epick_body.xacro and epick_single_suction_cup.xacro).
        # Rotate so local +Z aligns with UR tool-forward (+Y in current UR XML).
        epick_base = ET.SubElement(
            wrist3,
            "body",
            {
                "name": "epick_base_link",
                "pos": "0 0.0921 0",
                "quat": "0.70710678 -0.70710678 0 0",
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
                # Keep EPick body physically collidable with workspace/object geoms.
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

        # Lift target marker.
        goal_body = ET.SubElement(worldbody, "body", {"name": "lift_goal", "pos": "0.14 0 0.95"})
        ET.SubElement(
            goal_body,
            "site",
            {
                "name": "lift_goal_site",
                "type": "sphere",
                "size": "0.02",
                "rgba": "0.1 0.9 0.2 0.4",
            },
        )

        # Single lifted object.
        obj_body = ET.SubElement(worldbody, "body", {"name": "lift_obj", "pos": "0.14 0 0.80"})
        ET.SubElement(obj_body, "freejoint", {"name": "lift_obj_free"})
        ET.SubElement(
            obj_body,
            "geom",
            {
                "name": "lift_obj_geom",
                "type": "box",
                "size": "0.02 0.02 0.02",
                "rgba": "0.9 0.25 0.2 1",
                "friction": "1.1 0.1 0.03",
                "mass": "0.08",
            },
        )

        return ET.tostring(root, encoding="unicode")

    # ------------------------------------------------------------------ API
    @property
    def action_dim(self) -> int:
        """Action dimensionality: ``[dx, dy, dz, dyaw, suction]``."""
        return 5

    @property
    def observation_dim(self) -> int:
        """Observation dimensionality."""
        # qpos(6) + qvel(6) + cup_pos(3) + obj_pos(3) + obj_vel(3) + goal_pos(3)
        # + cup_to_obj(3) + obj_to_goal(3) + suction(1) + grasped(1) + last_action(5)
        return 37

    # ------------------------------------------------------------------ Reset
    def _table_top_z(self) -> float:
        table_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        if table_id >= 0:
            table_pos = self.model.geom_pos[table_id]
            table_size = self.model.geom_size[table_id]
            return float(table_pos[2] + table_size[2])
        return 0.74

    def _sample_object_spawn(self) -> np.ndarray:
        x_lo, x_hi, y_lo, y_hi = self.spawn_xy_bounds
        x = self._rng.uniform(x_lo, x_hi)
        y = self._rng.uniform(y_lo, y_hi)
        z = self._table_top_z() + self.obj_half[2] + 0.001
        return np.array([x, y, z], dtype=float)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment and return initial observation."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        self.step_id = 0
        self.suction_on = False
        self.grasped = False
        self._grasp_offset[:] = 0.0
        self._last_action[:] = 0.0

        for qi, (qpos_adr, dof_adr, act_id) in enumerate(
            zip(self._robot_qpos_ids, self.robot_dofs, self.robot_actuators)
        ):
            self.data.qpos[qpos_adr] = self.init_q[qi]
            self.data.qvel[dof_adr] = 0.0
            self.data.ctrl[act_id] = self.init_q[qi]

        self._last_targets = self.init_q.copy()

        self._spawn_pos = self._sample_object_spawn()
        qadr = int(self.model.jnt_qposadr[self.obj_free_id])
        vad = int(self.model.jnt_dofadr[self.obj_free_id])
        self.data.qpos[qadr: qadr + 3] = self._spawn_pos
        self.data.qpos[qadr + 3: qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.data.qvel[vad: vad + 6] = 0.0

        self._goal_pos = self._spawn_pos.copy()
        self._goal_pos[2] = self._spawn_pos[2] + self.lift_height
        if self.goal_body_id >= 0:
            self.model.body_pos[self.goal_body_id] = self._goal_pos

        mujoco.mj_forward(self.model, self.data)
        self._target_yaw = self._ee_yaw()
        for _ in range(self.settle_steps):
            for k, act_id in enumerate(self.robot_actuators):
                self.data.ctrl[act_id] = self._last_targets[k]
            mujoco.mj_step(self.model, self.data)

        return self._observe()

    # ------------------------------------------------------------------ IK + control
    def _ee_yaw(self) -> float:
        mat = self.data.site_xmat[self.ee_site].reshape(3, 3)
        return float(math.atan2(mat[1, 0], mat[0, 0]))

    def _suction_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.suction_site].copy()

    def _desired_ee(
        self, delta_xyz: np.ndarray, dyaw: float
    ) -> Tuple[np.ndarray, float]:
        pos = self.data.site_xpos[self.ee_site].copy() + delta_xyz
        pos[0] = float(np.clip(pos[0], -0.10, 0.35))
        pos[1] = float(np.clip(pos[1], -0.35, 0.35))
        pos[2] = float(np.clip(pos[2], self._table_top_z() + 0.04, 1.25))
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
        J = np.vstack([jacp[:, cols], jacr[2:3, cols]])
        lam = float(self.ik_damping)
        JJ = J @ J.T + (lam * lam) * np.eye(J.shape[0])
        return J.T @ np.linalg.solve(JJ, target)

    def _clamp_to_limits(self, targets: np.ndarray) -> np.ndarray:
        out = targets.copy()
        for k, jid in enumerate(self._robot_joint_ids):
            lo, hi = self.model.jnt_range[jid]
            if lo < hi:
                out[k] = float(np.clip(out[k], lo, hi))
        return out

    # ------------------------------------------------------------------ Suction
    def _apply_suction_weld(self) -> None:
        if not self.grasped:
            return
        qadr = int(self.model.jnt_qposadr[self.obj_free_id])
        vad = int(self.model.jnt_dofadr[self.obj_free_id])
        target_pos = self._suction_pos() + self._grasp_offset
        self.data.qpos[qadr: qadr + 3] = target_pos
        self.data.qvel[vad: vad + 6] = 0.0

    def _update_suction(self, suction_pressed: bool) -> float:
        reward = 0.0
        if not suction_pressed:
            self.suction_on = False
            self.grasped = False
            return reward

        self.suction_on = True
        if not self.grasped:
            obj_pos = self.data.xpos[self.obj_body_id]
            suction_pos = self._suction_pos()
            if np.linalg.norm(obj_pos - suction_pos) < self.suction_radius:
                self.grasped = True
                self._grasp_offset = obj_pos - suction_pos
                reward += 0.25

        self._apply_suction_weld()
        return reward

    # ------------------------------------------------------------------ Reward / obs
    def _compute_reward(self) -> Tuple[float, bool, Dict]:
        obj_pos = self.data.xpos[self.obj_body_id].copy()
        suction_pos = self._suction_pos()
        qvel_adr = int(self.model.jnt_dofadr[self.obj_free_id])
        obj_vel = self.data.qvel[qvel_adr: qvel_adr + 3]

        cup_obj_dist = float(np.linalg.norm(suction_pos - obj_pos))
        obj_goal_dist = float(np.linalg.norm(obj_pos - self._goal_pos))
        lifted = float(max(0.0, obj_pos[2] - self._spawn_pos[2]))

        reward = -0.002
        reward -= 0.8 * cup_obj_dist
        reward -= 1.8 * obj_goal_dist
        reward += 1.2 * lifted
        if self.suction_on:
            reward += 0.01
        if self.grasped:
            reward += 0.20

        success = bool(obj_pos[2] >= self._goal_pos[2] and np.linalg.norm(obj_vel) < 0.2)
        if success:
            reward += 5.0

        time_out = bool(self.time_limit > 0 and self.step_id >= self.time_limit)
        done = bool(success or time_out)
        info = {
            "success": success,
            "time_out": time_out,
            "grasped": bool(self.grasped),
            "suction_on": bool(self.suction_on),
            "cup_obj_dist": cup_obj_dist,
            "obj_goal_dist": obj_goal_dist,
            "object_height": float(obj_pos[2]),
            "goal_height": float(self._goal_pos[2]),
        }
        return float(reward), done, info

    def _observe(self) -> np.ndarray:
        qpos = self.data.qpos
        qvel = self.data.qvel
        obj_qadr = int(self.model.jnt_qposadr[self.obj_free_id])
        obj_vadr = int(self.model.jnt_dofadr[self.obj_free_id])

        joint_pos = np.array([qpos[qid] for qid in self._robot_qpos_ids], dtype=np.float32)
        joint_vel = np.array([qvel[did] for did in self.robot_dofs], dtype=np.float32)
        suction_pos = self._suction_pos().astype(np.float32)
        obj_pos = qpos[obj_qadr: obj_qadr + 3].astype(np.float32)
        obj_vel = qvel[obj_vadr: obj_vadr + 3].astype(np.float32)
        goal_pos = self._goal_pos.astype(np.float32)

        cup_to_obj = (obj_pos - suction_pos).astype(np.float32)
        obj_to_goal = (goal_pos - obj_pos).astype(np.float32)

        obs = np.concatenate(
            [
                joint_pos,
                joint_vel,
                suction_pos,
                obj_pos,
                obj_vel,
                goal_pos,
                cup_to_obj,
                obj_to_goal,
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
        suction_pressed = bool(act[4] > 0.0)

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

        reward = self._update_suction(suction_pressed)
        self._apply_suction_weld()
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
            self._apply_suction_weld()

        shaped, done, info = self._compute_reward()
        reward += shaped
        self.step_id += 1
        return StepResult(self._observe(), float(reward), done, info)

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if mode == "human":
            return None
        if mode == "rgb_array":
            self.renderer.update_scene(self.data, camera="top")
            return self.renderer.render()
        raise ValueError("mode must be 'human' or 'rgb_array'")

    def close(self) -> None:
        if hasattr(self.renderer, "close"):
            self.renderer.close()
        elif hasattr(self.renderer, "free"):
            self.renderer.free()
        self.renderer = None

    def sample_action(self) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)
