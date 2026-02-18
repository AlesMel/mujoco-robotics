"""Lift-suction task configuration objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from mujoco_robot.assets.configs import get_robot_config


@dataclass
class LiftSuctionTaskConfig:
    """High-level configuration for the lift-suction task."""

    model_path: str | None = None
    actuator_profile: str = "ur3e"
    time_limit: int = 300
    seed: int | None = None
    render: bool = False
    render_mode: str | None = None
    env_kwargs: Dict[str, Any] = field(default_factory=dict)


def make_ur3e_lift_suction_cfg() -> LiftSuctionTaskConfig:
    """Default UR3e lift-suction profile."""
    robot = get_robot_config("ur3e")
    return LiftSuctionTaskConfig(
        model_path=robot.model_path,
        actuator_profile="ur3e",
        time_limit=300,
    )


def make_ur5e_lift_suction_cfg() -> LiftSuctionTaskConfig:
    """UR5e lift-suction profile (for shared UR5e assets/actuators)."""
    robot = get_robot_config("ur5e")
    return LiftSuctionTaskConfig(
        model_path=robot.model_path,
        actuator_profile="ur5e",
        time_limit=300,
    )


def make_ur3e_suction_contact_cfg() -> LiftSuctionTaskConfig:
    """Primitive contact-stage profile for suction pretraining."""
    robot = get_robot_config("ur3e")
    return LiftSuctionTaskConfig(
        model_path=robot.model_path,
        actuator_profile="ur3e",
        time_limit=300,
        env_kwargs={
            "contact_success_hold_steps": 100,
            "contact_spawn_jitter_xy": 0.02,
        },
    )


def make_ur5e_suction_contact_cfg() -> LiftSuctionTaskConfig:
    """Primitive contact-stage profile for suction pretraining (UR5e)."""
    robot = get_robot_config("ur5e")
    return LiftSuctionTaskConfig(
        model_path=robot.model_path,
        actuator_profile="ur5e",
        time_limit=300,
        env_kwargs={
            "contact_success_hold_steps": 50,
            "contact_spawn_jitter_xy": 0.02,
        },
    )


_CFG_FACTORIES: dict[str, Callable[[], LiftSuctionTaskConfig]] = {
    "ur3e_lift_suction": make_ur3e_lift_suction_cfg,
    "ur5e_lift_suction": make_ur5e_lift_suction_cfg,
    "ur3e_suction_contact": make_ur3e_suction_contact_cfg,
    "ur5e_suction_contact": make_ur5e_suction_contact_cfg,
    # Alias to match the reach "dense_stable" naming style.
    "ur3e_lift_suction_dense_stable": make_ur3e_lift_suction_cfg,
    "ur3e_suction_contact_dense_stable": make_ur3e_suction_contact_cfg,
}


def get_lift_suction_cfg(name: str) -> LiftSuctionTaskConfig:
    """Build one named lift-suction config profile."""
    if name not in _CFG_FACTORIES:
        raise ValueError(
            f"Unknown lift_suction cfg '{name}'. Available: {sorted(_CFG_FACTORIES)}"
        )
    return _CFG_FACTORIES[name]()


def list_lift_suction_cfgs() -> tuple[str, ...]:
    """List available lift-suction config profile names."""
    return tuple(sorted(_CFG_FACTORIES.keys()))

