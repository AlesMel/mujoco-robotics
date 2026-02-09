"""Reusable actuator profiles and MuJoCo lookup helpers.

This module keeps actuator-related configuration separate from task logic,
similar to IsaacLab's articulation/actuator split.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import mujoco
import numpy as np


@dataclass(frozen=True)
class ActuatorConfig:
    """Declarative position-servo actuator profile for a robot."""

    joint_names: Tuple[str, ...]
    actuator_name_template: str = "{joint}_motor"
    default_kp: float = 400.0
    fallback_ctrlrange: Tuple[float, float] = (-math.pi, math.pi)

    def actuator_name(self, joint_name: str) -> str:
        """Return the actuator name that commands ``joint_name``."""
        return self.actuator_name_template.format(joint=joint_name)


@dataclass(frozen=True)
class ResolvedActuators:
    """Runtime-resolved MuJoCo ids for one actuator profile."""

    config: ActuatorConfig
    joint_ids: Tuple[int, ...]
    qpos_ids: Tuple[int, ...]
    dof_ids: Tuple[int, ...]
    actuator_ids: Tuple[int, ...]

    @property
    def joint_names(self) -> Tuple[str, ...]:
        return self.config.joint_names


_UR_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist1",
    "wrist2",
    "wrist3",
)

_UR_POSITION_SERVO = ActuatorConfig(joint_names=_UR_JOINTS)

# Robot -> actuator profile mapping. Multiple robots can reuse one profile.
ROBOT_ACTUATOR_CONFIGS: Dict[str, ActuatorConfig] = {
    "ur5e": _UR_POSITION_SERVO,
    "ur3e": _UR_POSITION_SERVO,
}


def get_robot_actuator_config(name: str) -> ActuatorConfig:
    """Look up actuator profile for a robot name."""
    if name not in ROBOT_ACTUATOR_CONFIGS:
        raise ValueError(
            f"Unknown actuator profile '{name}'. "
            f"Available: {list(ROBOT_ACTUATOR_CONFIGS)}"
        )
    return ROBOT_ACTUATOR_CONFIGS[name]


def resolve_actuators(
    model: mujoco.MjModel,
    config: ActuatorConfig,
) -> ResolvedActuators:
    """Resolve MuJoCo ids for all joints/actuators in ``config``."""
    joint_ids = []
    qpos_ids = []
    dof_ids = []
    actuator_ids = []

    for joint_name in config.joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint '{joint_name}' not found in model.")
        act_name = config.actuator_name(joint_name)
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name
        )
        if actuator_id < 0:
            raise ValueError(
                f"Actuator '{act_name}' not found in model for joint '{joint_name}'."
            )

        joint_ids.append(joint_id)
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
        dof_ids.append(int(model.jnt_dofadr[joint_id]))
        actuator_ids.append(actuator_id)

    return ResolvedActuators(
        config=config,
        joint_ids=tuple(joint_ids),
        qpos_ids=tuple(qpos_ids),
        dof_ids=tuple(dof_ids),
        actuator_ids=tuple(actuator_ids),
    )


def resolve_robot_actuators(
    model: mujoco.MjModel,
    robot_name: str,
) -> ResolvedActuators:
    """Resolve actuator ids for the given robot profile name."""
    return resolve_actuators(model, get_robot_actuator_config(robot_name))


def configure_position_actuators(
    model: mujoco.MjModel,
    resolved: ResolvedActuators,
    *,
    min_damping: float = 0.0,
    min_frictionloss: float = 0.0,
    kp: float | None = None,
) -> None:
    """Apply standard servo defaults for the resolved actuators."""
    if min_damping < 0.0:
        raise ValueError("min_damping must be non-negative.")
    if min_frictionloss < 0.0:
        raise ValueError("min_frictionloss must be non-negative.")

    servo_kp = resolved.config.default_kp if kp is None else float(kp)
    fallback_low, fallback_high = resolved.config.fallback_ctrlrange

    for dof_id in resolved.dof_ids:
        model.dof_damping[dof_id] = max(model.dof_damping[dof_id], min_damping)
        model.dof_frictionloss[dof_id] = max(
            model.dof_frictionloss[dof_id], min_frictionloss
        )

    for joint_id, actuator_id in zip(resolved.joint_ids, resolved.actuator_ids):
        low, high = model.jnt_range[joint_id]
        if low >= high:
            low, high = fallback_low, fallback_high
        model.actuator_ctrlrange[actuator_id] = np.array([low, high], dtype=float)
        if model.actuator_gainprm[actuator_id, 0] <= 0.0:
            model.actuator_gainprm[actuator_id, 0] = servo_kp


__all__ = [
    "ActuatorConfig",
    "ResolvedActuators",
    "ROBOT_ACTUATOR_CONFIGS",
    "get_robot_actuator_config",
    "resolve_actuators",
    "resolve_robot_actuators",
    "configure_position_actuators",
]
