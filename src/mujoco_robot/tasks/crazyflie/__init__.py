"""Crazyflie task entrypoints and config (hover + reach)."""

from mujoco_robot.tasks.crazyflie.config import (
    CrazyflieTaskConfig,
    get_crazyflie_cfg,
    list_crazyflie_cfgs,
    make_crazyflie_hover_cfg,
    make_crazyflie_hover_dense_stable_cfg,
    make_crazyflie_hover_flowdeck_cfg,
    make_crazyflie_reach_cfg,
    make_crazyflie_reach_dense_stable_cfg,
    make_crazyflie_obstacle_cfg,
    make_crazyflie_obstacle_dense_stable_cfg,
)
from mujoco_robot.tasks.crazyflie.factory import (
    make_crazyflie_hover_env,
    make_crazyflie_hover_gymnasium,
    make_crazyflie_reach_env,
    make_crazyflie_reach_gymnasium,
    make_crazyflie_obstacle_env,
    make_crazyflie_obstacle_gymnasium,
)
from mujoco_robot.tasks.crazyflie.crazyflie_env import (
    CrazyflieHoverEnv,
    CrazyflieHoverGymnasium,
)
from mujoco_robot.tasks.crazyflie.crazyflie_reach_env import (
    CrazyflieReachEnv,
    CrazyflieReachGymnasium,
)
from mujoco_robot.tasks.crazyflie.crazyflie_obstacle_env import (
    CrazyflieObstacleEnv,
    CrazyflieObstacleGymnasium,
)

__all__ = [
    "CrazyflieTaskConfig",
    "get_crazyflie_cfg",
    "list_crazyflie_cfgs",
    "make_crazyflie_hover_cfg",
    "make_crazyflie_hover_dense_stable_cfg",
    "make_crazyflie_hover_flowdeck_cfg",
    "make_crazyflie_reach_cfg",
    "make_crazyflie_reach_dense_stable_cfg",
    "CrazyflieHoverEnv",
    "CrazyflieHoverGymnasium",
    "CrazyflieReachEnv",
    "CrazyflieReachGymnasium",
    "make_crazyflie_hover_env",
    "make_crazyflie_hover_gymnasium",
    "make_crazyflie_reach_env",
    "make_crazyflie_reach_gymnasium",
    "make_crazyflie_obstacle_cfg",
    "make_crazyflie_obstacle_dense_stable_cfg",
    "CrazyflieObstacleEnv",
    "CrazyflieObstacleGymnasium",
    "make_crazyflie_obstacle_env",
    "make_crazyflie_obstacle_gymnasium",
]
