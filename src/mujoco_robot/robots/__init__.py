"""Robot definitions — model configs and MJCF assets."""

from mujoco_robot.robots.configs import ROBOT_CONFIGS, get_robot_config
from mujoco_robot.robots.actuators import (
    ActuatorConfig,
    ResolvedActuators,
    ROBOT_ACTUATOR_CONFIGS,
    get_robot_actuator_config,
    resolve_actuators,
    resolve_robot_actuators,
    configure_position_actuators,
)

__all__ = [
    "ROBOT_CONFIGS",
    "get_robot_config",
    "ActuatorConfig",
    "ResolvedActuators",
    "ROBOT_ACTUATOR_CONFIGS",
    "get_robot_actuator_config",
    "resolve_actuators",
    "resolve_robot_actuators",
    "configure_position_actuators",
]
