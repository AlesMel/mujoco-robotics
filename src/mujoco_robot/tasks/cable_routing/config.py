"""Cable-routing task configuration objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from mujoco_robot.assets.configs import get_robot_config


@dataclass
class CableRoutingTaskConfig:
    """High-level configuration for the cable-routing task."""

    model_path: str | None = None
    actuator_profile: str = "ur3e"
    time_limit: int = 450
    seed: int | None = None
    render: bool = False
    render_mode: str | None = None
    env_kwargs: Dict[str, Any] = field(default_factory=dict)


def make_ur3e_cable_routing_cfg() -> CableRoutingTaskConfig:
    """Default UR3e cable-routing profile."""
    robot = get_robot_config("ur3e")
    return CableRoutingTaskConfig(
        model_path=robot.model_path,
        actuator_profile="ur3e",
        time_limit=450,
    )


def make_ur5e_cable_routing_cfg() -> CableRoutingTaskConfig:
    """UR5e cable-routing profile (uses the same task assets)."""
    robot = get_robot_config("ur5e")
    return CableRoutingTaskConfig(
        model_path=robot.model_path,
        actuator_profile="ur5e",
        time_limit=450,
    )


_CFG_FACTORIES: dict[str, Callable[[], CableRoutingTaskConfig]] = {
    "ur3e_cable_routing": make_ur3e_cable_routing_cfg,
    "ur5e_cable_routing": make_ur5e_cable_routing_cfg,
    "ur3e_cable_routing_dense_stable": make_ur3e_cable_routing_cfg,
}


def get_cable_routing_cfg(name: str) -> CableRoutingTaskConfig:
    """Build one named cable-routing config profile."""
    if name not in _CFG_FACTORIES:
        raise ValueError(
            f"Unknown cable_routing cfg '{name}'. Available: {sorted(_CFG_FACTORIES)}"
        )
    return _CFG_FACTORIES[name]()


def list_cable_routing_cfgs() -> tuple[str, ...]:
    """List available cable-routing profile names."""
    return tuple(sorted(_CFG_FACTORIES.keys()))
