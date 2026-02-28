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


def make_ur3e_cable_routing_dense_stable_cfg() -> CableRoutingTaskConfig:
    robot = get_robot_config("ur3e")
    return CableRoutingTaskConfig(
        model_path=robot.model_path,
        actuator_profile="ur3e",
        time_limit=450,
        env_kwargs={
            "n_substeps": 5,
            "solver_iterations": 100,
            "solver_noslip_iterations": 20,
            "settle_steps": 500,
        },
    )


def make_ur3e_cable_routing_train_fast_cfg() -> CableRoutingTaskConfig:
    robot = get_robot_config("ur3e")
    return CableRoutingTaskConfig(
        model_path=robot.model_path,
        actuator_profile="ur3e",
        time_limit=350,
        env_kwargs={
            "n_substeps": 2,
            "solver_iterations": 60,
            "solver_noslip_iterations": 10,
            "settle_steps": 120,
        },
    )


def make_ur3e_cable_grasp_cfg() -> CableRoutingTaskConfig:
    """UR3e cable-grasp subtask: reach the cable free end and grasp it.

    Uses a dense distance-based reward (like the reach task).  Trains in
    ~300-500 k timesteps with PPO.  Once this policy works, load it as a
    starting point for the full routing task.
    """
    robot = get_robot_config("ur3e")
    return CableRoutingTaskConfig(
        model_path=robot.model_path,
        actuator_profile="ur3e",
        time_limit=120,
        env_kwargs={
            "task_mode": "grasp",
            "ee_step": 0.06,
            "n_substeps": 2,
            "solver_iterations": 60,
            "solver_noslip_iterations": 10,
            "settle_steps": 120,
            # Easier suction in grasp mode: deterministic, wider radius.
            "suction_attach_stochastic": False,
            "suction_radius": 0.035,
            "suction_alignment_cos_min": 0.30,
        },
    )


def make_ur3e_cable_route_1clip_cfg() -> CableRoutingTaskConfig:
    """Route cable through 1 clip only – intermediate difficulty.

    Start training from a pretrained cable-grasp checkpoint.
    """
    robot = get_robot_config("ur3e")
    return CableRoutingTaskConfig(
        model_path=robot.model_path,
        actuator_profile="ur3e",
        time_limit=200,
        env_kwargs={
            "task_mode": "route",
            "n_substeps": 2,
            "solver_iterations": 60,
            "solver_noslip_iterations": 10,
            "settle_steps": 120,
            "clip_positions": [[0.08, -0.030]],
            "suction_attach_stochastic": False,
            "suction_radius": 0.030,
            "suction_alignment_cos_min": 0.35,
        },
    )


_CFG_FACTORIES: dict[str, Callable[[], CableRoutingTaskConfig]] = {
    "ur3e_cable_routing": make_ur3e_cable_routing_cfg,
    "ur5e_cable_routing": make_ur5e_cable_routing_cfg,
    "ur3e_cable_routing_dense_stable": make_ur3e_cable_routing_dense_stable_cfg,
    "ur3e_cable_routing_train_fast": make_ur3e_cable_routing_train_fast_cfg,
    "ur3e_cable_grasp": make_ur3e_cable_grasp_cfg,
    "ur3e_cable_route_1clip": make_ur3e_cable_route_1clip_cfg,
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
