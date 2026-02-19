"""Cable-routing task entrypoints and config."""

from mujoco_robot.tasks.cable_routing.config import (
    CableRoutingTaskConfig,
    get_cable_routing_cfg,
    list_cable_routing_cfgs,
    make_ur3e_cable_routing_cfg,
    make_ur5e_cable_routing_cfg,
)
from mujoco_robot.tasks.cable_routing.factory import (
    make_cable_routing_env,
    make_cable_routing_gymnasium,
)
from mujoco_robot.tasks.cable_routing.cable_routing_env import (
    CableRoutingGymnasium,
    URCableRoutingEnv,
)

__all__ = [
    "CableRoutingTaskConfig",
    "get_cable_routing_cfg",
    "list_cable_routing_cfgs",
    "make_ur3e_cable_routing_cfg",
    "make_ur5e_cable_routing_cfg",
    "CableRoutingGymnasium",
    "URCableRoutingEnv",
    "make_cable_routing_env",
    "make_cable_routing_gymnasium",
]
