"""Crazyflie 2.1 hover environment with motor-level control."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import gymnasium
import mujoco
import numpy as np

from mujoco_robot.core.xml_builder import load_robot_xml, set_framebuffer_size
from mujoco_robot.envs.step_result import StepResult


_DEFAULT_MODEL = str(
    Path(__file__).resolve().parent.parent.parent / "assets" / "crazyflie_2_1.xml"
)


def _quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build quaternion in MuJoCo [w, x, y, z] order."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )


class CrazyflieHoverGymnasium(gymnasium.Env):
    """Gymnasium wrapper around :class:`CrazyflieHoverEnv`."""

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        seed: int | None = None,
        render: bool = False,
        render_mode: str | None = None,
        time_limit: int = 600,
        model_path: str = _DEFAULT_MODEL,
        actuator_profile: str = "crazyflie",
        **env_kwargs,
    ):
        resolved_render_mode = render_mode or ("rgb_array" if render else None)
        if resolved_render_mode not in {None, "rgb_array", "human"}:
            raise ValueError("render_mode must be one of: None, 'rgb_array', 'human'")

        self.base = CrazyflieHoverEnv(
            model_path=model_path,
            actuator_profile=actuator_profile,
            render_size=(640, 480) if resolved_render_mode == "rgb_array" else (240, 180),
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
        terminated = bool(
            res.info.get("success", False)
            or res.info.get("crashed", False)
            or res.info.get("battery_depleted", False)
            or res.info.get("out_of_bounds", False)
            or res.info.get("excessive_tilt", False)
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
                except Exception as exc:  # pragma: no cover - GUI availability is host-specific.
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


class CrazyflieHoverEnv:
    """Motor-level Crazyflie 2.1 hover task."""

    VALID_OBSERVATION_MODES = ("state_estimate", "flowdeck", "privileged")

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        actuator_profile: str = "crazyflie",
        time_limit: int = 600,
        observation_mode: str = "state_estimate",
        n_substeps: int = 10,
        render_size: Tuple[int, int] = (960, 720),
        seed: Optional[int] = None,
        max_motor_omega: float = 2600.0,
        tau_motor: float = 0.02,
        thrust_coeff: float = 2.45e-8,
        torque_coeff: float = 1.46e-10,
        arm_length: float = 0.03253,
        ground_effect_height: float = 0.092,
        ground_effect_gain: float = 0.25,
        drag_coeff_lin: Tuple[float, float, float] = (0.08, 0.08, 0.12),
        drag_coeff_ang: Tuple[float, float, float] = (2.0e-6, 2.0e-6, 4.0e-6),
        actuator_noise_std: float = 0.015,
        sensor_noise_scale: float = 1.0,
        disturbance_theta: float = 1.5,
        disturbance_sigma: float = 0.03,
        disturbance_max: float = 0.12,
        disturbance_torque_fraction: float = 0.0,
        battery_capacity_mah: float = 250.0,
        battery_nominal_voltage: float = 3.7,
        use_mixer: bool = False,
    ) -> None:
        self._rng = np.random.default_rng(seed)

        if actuator_profile != "crazyflie":
            raise ValueError(
                "Crazyflie hover environment expects actuator_profile='crazyflie'."
            )
        if observation_mode not in self.VALID_OBSERVATION_MODES:
            raise ValueError(
                f"observation_mode must be one of {self.VALID_OBSERVATION_MODES}, "
                f"got '{observation_mode}'."
            )

        self.model_path = model_path
        self.actuator_profile = actuator_profile
        self.observation_mode = observation_mode
        self.time_limit = int(max(1, time_limit))
        self.n_substeps = int(max(1, n_substeps))
        self.render_size = render_size

        self.max_motor_omega = float(max_motor_omega)
        self.tau_motor = float(max(1e-6, tau_motor))
        self.thrust_coeff = float(thrust_coeff)
        self.torque_coeff = float(torque_coeff)
        self.arm_length = float(arm_length)

        self.ground_effect_height = float(max(1e-4, ground_effect_height))
        self.ground_effect_gain = float(max(0.0, ground_effect_gain))

        self.drag_coeff_lin = np.asarray(drag_coeff_lin, dtype=float)
        self.drag_coeff_ang = np.asarray(drag_coeff_ang, dtype=float)

        self.actuator_noise_std = float(max(0.0, actuator_noise_std))
        self.sensor_noise_scale = float(max(0.0, sensor_noise_scale))

        self.disturbance_theta = float(max(0.0, disturbance_theta))
        self.disturbance_sigma = float(max(0.0, disturbance_sigma))
        self.disturbance_max = float(max(0.0, disturbance_max))
        self.disturbance_torque_fraction = float(max(0.0, disturbance_torque_fraction))

        self.battery_capacity_mah = float(max(1e-6, battery_capacity_mah))
        self.battery_nominal_voltage = float(max(1e-6, battery_nominal_voltage))
        self.battery_energy_j = (
            self.battery_capacity_mah / 1000.0 * self.battery_nominal_voltage * 3600.0
        )

        # Optional collective/attitude mixer (False = direct per-motor control)
        self.use_mixer = bool(use_mixer)

        self.hover_success_pos_tol = 0.05
        self.hover_success_tilt_deg = 8.0
        self.hover_success_speed_tol = 0.2
        self.hover_success_hold_steps = 40
        self.termination_tilt_deg = 75.0

        self.workspace_xy = 1.0
        self.workspace_z = (0.05, 1.5)
        self.goal_xy = 0.18
        self.goal_z = (0.22, 0.55)

        self.motor_dirs = np.array([1.0, -1.0, 1.0, -1.0], dtype=float)
        self.motor_offsets_body = np.array(
            [
                [self.arm_length, self.arm_length, 0.0],
                [-self.arm_length, self.arm_length, 0.0],
                [-self.arm_length, -self.arm_length, 0.0],
                [self.arm_length, -self.arm_length, 0.0],
            ],
            dtype=float,
        )

        model_xml = self._load_model_xml(model_path)
        self.model = mujoco.MjModel.from_xml_string(model_xml)
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = 0.001
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

        self.dt_control = float(self.model.opt.timestep) * float(self.n_substeps)

        self.cf_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cf2")
        self.cf_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cf2_free")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if self.cf_body_id < 0 or self.cf_joint_id < 0:
            raise ValueError("Crazyflie model is missing required 'cf2' body/freejoint.")

        self._cf_qpos_adr = int(self.model.jnt_qposadr[self.cf_joint_id])
        self._cf_qvel_adr = int(self.model.jnt_dofadr[self.cf_joint_id])

        self.motor_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"m{i}_joint")
            for i in range(1, 5)
        ]
        # DOF addresses for direct qvel writes (no MuJoCo actuators needed)
        self._rotor_dof_adrs = [
            int(self.model.jnt_dofadr[jid]) for jid in self.motor_joint_ids
        ]
        self.motor_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"m{i}_site")
            for i in range(1, 5)
        ]
        if any(jid < 0 for jid in self.motor_joint_ids):
            raise ValueError("Crazyflie model must define joints m1_joint..m4_joint.")
        if any(mid < 0 for mid in self.motor_site_ids):
            raise ValueError("Crazyflie model must define sites m1_site..m4_site.")

        self._sensor_slices = self._resolve_sensor_slices(
            [
                "imu_gyro",
                "imu_accel",
                "cf_pos",
                "cf_linvel",
                "cf_angvel",
                "cf_quat",
            ]
        )

        self._drone_geom_ids = self._collect_drone_geom_ids()

        self._camera_names = ["cf_chase", "cf_top", "cf_side", "cf_closeup"]

        rw, rh = self.render_size
        self._renderers = [
            mujoco.Renderer(self.model, height=rh, width=rw)
            for _ in self._camera_names
        ]

        self.step_id = 0
        self._goal_pos = np.zeros(3, dtype=float)
        self._goal_yaw = 0.0
        self._success_hold_counter = 0

        self.motor_omega = np.zeros(4, dtype=float)
        self._last_action = np.zeros(4, dtype=np.float32)
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._last_motor_cmd_norm = np.zeros(4, dtype=float)

        self.battery_soc = 1.0
        self._disturbance_force_world = np.zeros(3, dtype=float)
        self._disturbance_torque_body = np.zeros(3, dtype=float)

        self._last_reward = 0.0
        self._last_ground_effect_mean = 1.0
        self._last_info: Dict[str, float | bool | int] = {}

    # ------------------------------------------------------------------ Setup
    def _load_model_xml(self, model_path: str) -> str:
        xml_text = load_robot_xml(model_path)
        root = ET.fromstring(xml_text)
        set_framebuffer_size(root, self.render_size[0], self.render_size[1])
        return ET.tostring(root, encoding="unicode")

    def _resolve_sensor_slices(self, sensor_names: list[str]) -> Dict[str, slice]:
        out: Dict[str, slice] = {}
        for name in sensor_names:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid < 0:
                raise ValueError(f"Crazyflie model missing required sensor '{name}'.")
            adr = int(self.model.sensor_adr[sid])
            dim = int(self.model.sensor_dim[sid])
            out[name] = slice(adr, adr + dim)
        return out

    def _collect_drone_geom_ids(self) -> set[int]:
        body_names = {
            "cf2",
            "cf2_battery",
            "m1_rotor",
            "m2_rotor",
            "m3_rotor",
            "m4_rotor",
        }
        body_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in body_names
        }
        body_ids = {bid for bid in body_ids if bid >= 0}

        drone_geom_ids: set[int] = set()
        for gid in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[gid])
            if body_id not in body_ids:
                continue
            if int(self.model.geom_contype[gid]) <= 0 and int(self.model.geom_conaffinity[gid]) <= 0:
                continue
            drone_geom_ids.add(gid)
        return drone_geom_ids

    # ------------------------------------------------------------------ API
    @property
    def action_dim(self) -> int:
        return 4

    @property
    def observation_dim(self) -> int:
        base = 3 + 3 + 4 + 3 + 3 + 1 + 4 + 1 + 4
        if self.observation_mode == "flowdeck":
            return base + 2
        return base

    # ------------------------------------------------------------------ Dynamics helpers
    def _sensor(self, name: str) -> np.ndarray:
        sl = self._sensor_slices[name]
        return self.data.sensordata[sl].copy()

    def _quat_yaw(self, quat_wxyz: np.ndarray) -> float:
        w, x, y, z = quat_wxyz
        s3 = 2.0 * (w * z + x * y)
        c3 = 1.0 - 2.0 * (y * y + z * z)
        return float(math.atan2(s3, c3))

    def _body_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos = self.data.xpos[self.cf_body_id].copy()
        quat = self.data.xquat[self.cf_body_id].copy()
        cvel = self.data.cvel[self.cf_body_id]
        ang_vel_world = cvel[:3].copy()
        lin_vel_world = cvel[3:].copy()
        rot_mat = self.data.xmat[self.cf_body_id].reshape(3, 3).copy()
        return pos, quat, lin_vel_world, ang_vel_world, rot_mat

    def _sample_goal(self) -> tuple[np.ndarray, float]:
        pos = np.array(
            [
                self._rng.uniform(-self.goal_xy, self.goal_xy),
                self._rng.uniform(-self.goal_xy, self.goal_xy),
                self._rng.uniform(self.goal_z[0], self.goal_z[1]),
            ],
            dtype=float,
        )
        yaw = float(self._rng.uniform(-math.pi, math.pi))
        return pos, yaw

    def _ground_effect_multiplier(
        self, rotor_height: float, cos_tilt: float, horiz_speed: float,
    ) -> float:
        """Height-dependent ground-effect with tilt & lateral-velocity attenuation.

        When the drone is tilted the rotor wash is deflected sideways, reducing
        the ground-effect cushion.  Similarly, lateral motion sweeps the
        recirculation zone away.  Both factors reduce the multiplier toward 1.0.
        """
        if rotor_height >= self.ground_effect_height:
            return 1.0
        ratio = 1.0 - (rotor_height / self.ground_effect_height)
        base_mult = self.ground_effect_gain * ratio * ratio

        # Tilt attenuation: full effect only when body Z is vertical
        tilt_factor = max(cos_tilt, 0.0) ** 2  # cos²(tilt)

        # Lateral-velocity attenuation (exponential decay, ~0.5 m/s half-life)
        speed_factor = math.exp(-1.4 * horiz_speed)

        mult = 1.0 + base_mult * tilt_factor * speed_factor
        return float(np.clip(mult, 1.0, 1.25))

    def _update_disturbance(self) -> None:
        """Advance Ornstein-Uhlenbeck processes for wind force *and* torque."""
        if self.disturbance_sigma <= 0.0:
            self._disturbance_force_world[:] = 0.0
            self._disturbance_torque_body[:] = 0.0
            return

        dt = self.dt_control
        sqrt_dt = math.sqrt(dt)

        # --- Force (world frame) ---
        noise_f = self._rng.normal(0.0, 1.0, size=3)
        self._disturbance_force_world += (
            self.disturbance_theta * (-self._disturbance_force_world) * dt
            + self.disturbance_sigma * sqrt_dt * noise_f
        )
        nrm = float(np.linalg.norm(self._disturbance_force_world))
        if nrm > self.disturbance_max and nrm > 1e-9:
            self._disturbance_force_world *= self.disturbance_max / nrm

        # --- Torque (body frame) ---
        # Gust-induced moments from asymmetric pressure on the airframe.
        torque_sigma = self.disturbance_sigma * self.disturbance_torque_fraction
        torque_max = self.disturbance_max * self.disturbance_torque_fraction
        noise_t = self._rng.normal(0.0, 1.0, size=3)
        self._disturbance_torque_body += (
            self.disturbance_theta * (-self._disturbance_torque_body) * dt
            + torque_sigma * sqrt_dt * noise_t
        )
        nrm_t = float(np.linalg.norm(self._disturbance_torque_body))
        if nrm_t > torque_max and nrm_t > 1e-9:
            self._disturbance_torque_body *= torque_max / nrm_t

    def _compute_external_wrenches(self) -> tuple[np.ndarray, np.ndarray, float]:
        _pos, _quat, lin_vel_world, ang_vel_world, rot_mat = self._body_state()

        body_z_world = rot_mat[:, 2]
        cos_tilt = float(body_z_world[2])  # dot(body_z, world_z)
        horiz_speed = float(np.linalg.norm(lin_vel_world[:2]))
        lin_vel_body = rot_mat.T @ lin_vel_world
        ang_vel_body = rot_mat.T @ ang_vel_world

        total_force_world = np.zeros(3, dtype=float)
        total_torque_body = np.zeros(3, dtype=float)
        ground_mults = np.ones(4, dtype=float)

        for i in range(4):
            rotor_height = float(self.data.site_xpos[self.motor_site_ids[i]][2])
            ge_mult = self._ground_effect_multiplier(rotor_height, cos_tilt, horiz_speed)
            ground_mults[i] = ge_mult
            thrust = self.thrust_coeff * (self.motor_omega[i] ** 2) * ge_mult

            force_body = np.array([0.0, 0.0, thrust], dtype=float)
            force_world = body_z_world * thrust

            total_force_world += force_world
            total_torque_body += np.cross(self.motor_offsets_body[i], force_body)
            total_torque_body[2] += self.motor_dirs[i] * self.torque_coeff * (
                self.motor_omega[i] ** 2
            )

        drag_force_body = -self.drag_coeff_lin * np.abs(lin_vel_body) * lin_vel_body
        drag_torque_body = -self.drag_coeff_ang * np.abs(ang_vel_body) * ang_vel_body

        total_force_world += rot_mat @ drag_force_body
        total_force_world += self._disturbance_force_world
        total_torque_body += drag_torque_body
        total_torque_body += self._disturbance_torque_body

        total_torque_world = rot_mat @ total_torque_body
        ground_effect_mean = float(np.mean(ground_mults))
        return total_force_world, total_torque_world, ground_effect_mean

    # --- Mixer matrix (collective + roll/pitch/yaw → per-motor) ---
    _MIXER = np.array(
        [
            [1.0,  1.0,  1.0,  1.0],   # collective (front-right, back-left CW)
            [1.0, -1.0, -1.0,  1.0],   # roll
            [1.0,  1.0, -1.0, -1.0],   # pitch
            [1.0, -1.0,  1.0, -1.0],   # yaw (reaction torque signs)
        ],
        dtype=float,
    ).T  # shape (4 motors, 4 channels)

    def _apply_motor_and_battery_dynamics(self, action: np.ndarray) -> None:
        # --- Action mapping ---
        if self.use_mixer:
            # action = [collective, roll, pitch, yaw] in [-1, 1]
            collective = (action[0] + 1.0) * 0.5  # → [0, 1]
            att = action[1:4] * 0.5               # → [-0.5, 0.5]
            mixed = np.array([collective, att[0], att[1], att[2]], dtype=float)
            cmd_norm = np.clip(self._MIXER @ mixed, 0.0, 1.0)
        else:
            cmd_norm = np.clip((action + 1.0) * 0.5, 0.0, 1.0)

        if self.actuator_noise_std > 0.0:
            cmd_norm = cmd_norm + self._rng.normal(0.0, self.actuator_noise_std, size=4)
        cmd_norm = np.clip(cmd_norm, 0.0, 1.0)

        # Voltage sag — simple LiPo discharge curve
        voltage_scale = float(np.clip(0.78 + 0.22 * self.battery_soc, 0.0, 1.0))
        omega_des = cmd_norm * self.max_motor_omega * voltage_scale

        alpha = self.dt_control / self.tau_motor
        self.motor_omega += alpha * (omega_des - self.motor_omega)
        self.motor_omega = np.clip(self.motor_omega, 0.0, self.max_motor_omega * voltage_scale)

        self._last_motor_cmd_norm = cmd_norm

        # --- Thrust-based battery drain (calibrated to real CF2.1) ---
        # ~6.5 W at hover (~63 % throttle), ~12 W at full throttle
        omega_frac = self.motor_omega / self.max_motor_omega
        power = 0.5 + float(np.sum(2.0 * omega_frac ** 2 + 1.0 * omega_frac ** 3))
        self.battery_soc -= (power * self.dt_control) / self.battery_energy_j
        self.battery_soc = float(np.clip(self.battery_soc, 0.0, 1.0))

    # ------------------------------------------------------------------ Reset/Observe
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.step_id = 0
        self._success_hold_counter = 0
        self._prev_action[:] = 0.0
        self._last_action[:] = 0.0

        self._goal_pos, self._goal_yaw = self._sample_goal()

        spawn_pos = self._goal_pos.copy()
        spawn_pos[:2] += self._rng.normal(0.0, 0.03, size=2)
        spawn_pos[2] += float(self._rng.normal(0.0, 0.02))
        spawn_pos[2] = float(np.clip(spawn_pos[2], 0.18, 0.65))

        roll0 = float(self._rng.normal(0.0, math.radians(2.0)))
        pitch0 = float(self._rng.normal(0.0, math.radians(2.0)))
        yaw0 = float(self._goal_yaw + self._rng.normal(0.0, math.radians(7.0)))
        quat0 = _quat_from_euler(roll0, pitch0, yaw0)

        self.data.qpos[self._cf_qpos_adr: self._cf_qpos_adr + 3] = spawn_pos
        self.data.qpos[self._cf_qpos_adr + 3: self._cf_qpos_adr + 7] = quat0
        self.data.qvel[self._cf_qvel_adr: self._cf_qvel_adr + 6] = 0.0

        # Start motors near hover speed so the drone doesn't free-fall
        total_mass = 0.027  # cf2 (0.021) + battery (0.006) + rotors (~0)
        hover_omega = float(
            np.sqrt(total_mass * 9.81 / (4.0 * self.thrust_coeff))
        )
        self.motor_omega[:] = hover_omega * (
            1.0 + self._rng.normal(0.0, 0.02, size=4)
        )
        self.motor_omega[:] = np.clip(self.motor_omega, 0.0, self.max_motor_omega)
        for i, dof in enumerate(self._rotor_dof_adrs):
            self.data.qvel[dof] = self.motor_omega[i]

        self.battery_soc = float(self._rng.uniform(0.95, 1.0))
        self._disturbance_force_world[:] = 0.0
        self._disturbance_torque_body[:] = 0.0
        self._last_ground_effect_mean = 1.0
        self._last_reward = 0.0
        self._last_info = {}

        mujoco.mj_forward(self.model, self.data)
        return self._observe()

    def _observe(self) -> np.ndarray:
        pos, quat, lin_vel_world, _ang_vel_world, rot_mat = self._body_state()

        pos_err = (self._goal_pos - pos).astype(np.float32)

        if self.observation_mode == "privileged":
            gyro = self._sensor("cf_angvel").astype(np.float32)
            accel = self._sensor("imu_accel").astype(np.float32)
            baro_alt = np.array([pos[2]], dtype=np.float32)
            vel_world = lin_vel_world.astype(np.float32)
        else:
            gyro = self._sensor("imu_gyro")
            accel = self._sensor("imu_accel")
            vel_world = self._sensor("cf_linvel")
            baro_alt = np.array([pos[2]], dtype=float)

            nscale = self.sensor_noise_scale
            if nscale > 0.0:
                gyro += self._rng.normal(0.0, 0.02 * nscale, size=3)
                accel += self._rng.normal(0.0, 0.08 * nscale, size=3)
                vel_world += self._rng.normal(0.0, 0.015 * nscale, size=3)
                baro_alt += self._rng.normal(0.0, 0.01 * nscale, size=1)

            gyro = gyro.astype(np.float32)
            accel = accel.astype(np.float32)
            vel_world = vel_world.astype(np.float32)
            baro_alt = baro_alt.astype(np.float32)

        motor_norm = (self.motor_omega / self.max_motor_omega).astype(np.float32)

        obs_parts = [
            pos_err,
            vel_world,
            quat.astype(np.float32),
            gyro,
            accel,
            baro_alt,
            motor_norm,
            np.array([self.battery_soc], dtype=np.float32),
            self._last_action.astype(np.float32),
        ]

        if self.observation_mode == "flowdeck":
            vel_body = rot_mat.T @ lin_vel_world
            flow_xy = vel_body[:2]
            if self.sensor_noise_scale > 0.0:
                flow_xy = flow_xy + self._rng.normal(
                    0.0, 0.02 * self.sensor_noise_scale, size=2
                )
            obs_parts.append(flow_xy.astype(np.float32))

        return np.concatenate(obs_parts).astype(np.float32)

    # ------------------------------------------------------------------ Reward/Termination
    def _compute_reward_and_info(self) -> tuple[float, bool, Dict[str, float | bool | int]]:
        pos, quat, lin_vel_world, ang_vel_world, rot_mat = self._body_state()

        pos_error = float(np.linalg.norm(self._goal_pos - pos))
        lin_speed = float(np.linalg.norm(lin_vel_world))
        ang_speed = float(np.linalg.norm(ang_vel_world))

        body_z = rot_mat[:, 2]
        tilt_rad = float(math.acos(np.clip(np.dot(body_z, np.array([0.0, 0.0, 1.0])), -1.0, 1.0)))
        tilt_deg = math.degrees(tilt_rad)

        yaw = self._quat_yaw(quat)
        yaw_err = ((self._goal_yaw - yaw + math.pi) % (2 * math.pi)) - math.pi

        stable = bool(
            pos_error < self.hover_success_pos_tol
            and tilt_deg < self.hover_success_tilt_deg
            and lin_speed < self.hover_success_speed_tol
        )
        if stable:
            self._success_hold_counter += 1
        else:
            self._success_hold_counter = 0

        success = bool(self._success_hold_counter >= self.hover_success_hold_steps)

        crashed = bool(self._is_crashed_with_floor() or pos[2] <= self.workspace_z[0])
        out_of_bounds = bool(
            abs(pos[0]) > self.workspace_xy
            or abs(pos[1]) > self.workspace_xy
            or pos[2] > self.workspace_z[1]
        )
        excessive_tilt = bool(tilt_deg > self.termination_tilt_deg)
        battery_depleted = bool(self.battery_soc <= 0.05)
        time_out = bool(self.step_id >= self.time_limit)

        action_rate = float(np.linalg.norm(self._last_action - self._prev_action))

        reward = 0.03
        reward += 0.9 * math.exp(-pos_error / 0.08)
        reward -= 1.5 * pos_error
        reward -= 0.25 * abs(yaw_err)
        reward -= 0.18 * tilt_rad
        reward -= 0.15 * lin_speed
        reward -= 0.05 * ang_speed
        reward -= 0.02 * action_rate

        if stable:
            reward += 0.06
        if success:
            reward += 3.0
        if crashed:
            reward -= 3.0
        if battery_depleted:
            reward -= 1.0

        done = bool(success or crashed or out_of_bounds or excessive_tilt or battery_depleted or time_out)

        info: Dict[str, float | bool | int] = {
            "success": success,
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
            "success_hold_steps": int(self._success_hold_counter),
            "success_hold_required": int(self.hover_success_hold_steps),
        }
        return float(reward), done, info

    def _is_crashed_with_floor(self) -> bool:
        if self.floor_geom_id < 0:
            return False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            if g1 == self.floor_geom_id and g2 in self._drone_geom_ids:
                return True
            if g2 == self.floor_geom_id and g1 in self._drone_geom_ids:
                return True
        return False

    # ------------------------------------------------------------------ Step/Render
    def step(self, action: Iterable[float]) -> StepResult:
        act = np.asarray(action, dtype=float).flatten()
        if act.shape[0] != self.action_dim:
            raise ValueError(f"action should have shape ({self.action_dim},)")
        act = np.clip(act, -1.0, 1.0)
        self._prev_action = self._last_action.copy()
        self._last_action = act.astype(np.float32)

        self._update_disturbance()
        self._apply_motor_and_battery_dynamics(act)

        for _ in range(self.n_substeps):
            force_world, torque_world, ge_mean = self._compute_external_wrenches()
            self._last_ground_effect_mean = ge_mean

            self.data.xfrc_applied[self.cf_body_id, :3] = force_world
            self.data.xfrc_applied[self.cf_body_id, 3:] = torque_world
            # Pin rotor qvel each substep to prevent gyroscopic drift
            for i, dof in enumerate(self._rotor_dof_adrs):
                self.data.qvel[dof] = self.motor_omega[i]
            mujoco.mj_step(self.model, self.data)
            self.data.xfrc_applied[self.cf_body_id, :] = 0.0

        self.step_id += 1
        reward, done, info = self._compute_reward_and_info()
        self._last_reward = reward
        self._last_info = dict(info)
        return StepResult(obs=self._observe(), reward=reward, done=done, info=info)

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if mode == "human":
            return None
        if mode != "rgb_array":
            raise ValueError("mode must be 'human' or 'rgb_array'")

        frames = []
        for renderer, cam_name in zip(self._renderers, self._camera_names):
            renderer.update_scene(self.data, camera=cam_name)
            frames.append(renderer.render())

        # Build a 2x2 grid: [chase | top] / [side | metrics]
        top_row = np.concatenate([frames[0], frames[1]], axis=1)

        # Draw a metrics overlay in place of the 4th camera
        metrics_panel = self._draw_metrics_overlay(frames[3].shape)
        bottom_row = np.concatenate([frames[2], metrics_panel], axis=1)

        return np.concatenate([top_row, bottom_row], axis=0)

    def _draw_metrics_overlay(self, shape: tuple) -> np.ndarray:
        """Render a dark panel with key flight metrics burned in."""
        h, w = shape[0], shape[1]
        panel = np.full((h, w, 3), 20, dtype=np.uint8)

        info = self._last_info
        if not info:
            return panel

        lines = [
            f"Step: {self.step_id:>5d} / {self.time_limit}",
            f"Pos err:  {info.get('pos_error', 0.0):>7.3f} m",
            f"Tilt:     {info.get('tilt_deg', 0.0):>7.1f} deg",
            f"Lin spd:  {info.get('lin_speed', 0.0):>7.3f} m/s",
            f"Ang spd:  {info.get('ang_speed', 0.0):>7.3f} rad/s",
            f"Battery:  {info.get('battery_soc', 1.0) * 100:>6.1f} %",
            f"Motor avg:{info.get('motor_omega_mean', 0.0):>7.0f} rad/s",
            f"GE mult:  {info.get('ground_effect_mean', 1.0):>7.3f}",
            f"Wind:     {info.get('disturbance_norm', 0.0):>7.4f} N",
            f"Reward:   {self._last_reward:>+7.3f}",
            f"Hold:     {info.get('success_hold_steps', 0):>3d}"
            f" / {info.get('success_hold_required', 0)}",
        ]

        # Status indicator
        if info.get("success"):
            lines.append("  >> SUCCESS <<")
        elif info.get("crashed"):
            lines.append("  >> CRASHED <<")
        elif info.get("excessive_tilt"):
            lines.append("  >> TILT FAIL <<")
        elif info.get("out_of_bounds"):
            lines.append("  >> OOB <<")
        elif info.get("battery_depleted"):
            lines.append("  >> BATTERY <<")

        # Burn text using a simple 1-pixel font rasteriser
        self._burn_text_lines(panel, lines, x=10, y=14, scale=1)
        return panel

    @staticmethod
    def _burn_text_lines(
        img: np.ndarray,
        lines: list[str],
        x: int = 8,
        y: int = 12,
        scale: int = 1,
        color: tuple = (220, 230, 240),
        line_height: int = 18,
    ) -> None:
        """Burn monospaced text into an RGB image (no PIL/OpenCV needed).

        Uses a tiny built-in 5x7 bitmap font. *scale* multiplies pixel size.
        """
        _FONT: dict[str, list[int]] = {
            " ": [0, 0, 0, 0, 0, 0, 0],
            "0": [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
            "1": [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
            "2": [0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F],
            "3": [0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E],
            "4": [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
            "5": [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
            "6": [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
            "7": [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
            "8": [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
            "9": [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
            ".": [0, 0, 0, 0, 0, 0, 0x04],
            ",": [0, 0, 0, 0, 0, 0x04, 0x08],
            ":": [0, 0, 0x04, 0, 0, 0x04, 0],
            "/": [0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10],
            "+": [0, 0, 0x04, 0x0E, 0x04, 0, 0],
            "-": [0, 0, 0, 0x0E, 0, 0, 0],
            "%": [0x11, 0x02, 0x02, 0x04, 0x08, 0x08, 0x11],
            ">": [0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08],
            "<": [0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02],
            "=": [0, 0, 0x1F, 0, 0x1F, 0, 0],
        }
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            _FONT[ch.lower()] = _FONT.get(ch.lower(), [0]*7)
        # Minimal uppercase letters
        _FONT.update({
            "A": [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
            "B": [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
            "C": [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
            "D": [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
            "E": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
            "F": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
            "G": [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
            "H": [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
            "I": [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
            "L": [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
            "M": [0x11, 0x1B, 0x15, 0x11, 0x11, 0x11, 0x11],
            "N": [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
            "O": [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
            "P": [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
            "R": [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
            "S": [0x0E, 0x11, 0x10, 0x0E, 0x01, 0x11, 0x0E],
            "T": [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
            "U": [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
            "W": [0x11, 0x11, 0x11, 0x11, 0x15, 0x1B, 0x11],
        })
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            if ch not in _FONT:
                _FONT[ch] = [0]*7
            _FONT[ch.lower()] = _FONT.get(ch.lower(), _FONT.get(ch, [0]*7))

        h, w = img.shape[:2]
        cy = y
        for line in lines:
            cx = x
            for ch in line:
                glyph = _FONT.get(ch, _FONT.get(ch.upper(), [0]*7))
                for row_i, row_bits in enumerate(glyph):
                    for col_i in range(5):
                        if row_bits & (0x10 >> col_i):
                            for dy in range(scale):
                                for dx in range(scale):
                                    py = cy + row_i * scale + dy
                                    px = cx + col_i * scale + dx
                                    if 0 <= py < h and 0 <= px < w:
                                        img[py, px] = color
                cx += 6 * scale
            cy += line_height * scale

    def close(self) -> None:
        for renderer in self._renderers:
            if renderer is not None:
                if hasattr(renderer, "close"):
                    renderer.close()
                elif hasattr(renderer, "free"):
                    renderer.free()
        self._renderers = []

    def sample_action(self) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)
