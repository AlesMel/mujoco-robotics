"""Generic manager-based environment runtime.

Provides the base classes for building config-driven RL environments:

* :class:`ManagerBasedEnv` — abstract base with reset/step/close
* :class:`ManagerBasedRLEnv` — Gymnasium-compatible base
* :class:`ManagerRuntime` — named manager container
"""

from mujoco_robot.envs.base_env import ManagerBasedEnv
from mujoco_robot.envs.base_rl_env import ManagerBasedRLEnv
from mujoco_robot.envs.manager_runtime import ManagerRuntime
from mujoco_robot.envs.step_result import StepResult

__all__ = ["ManagerBasedEnv", "ManagerBasedRLEnv", "ManagerRuntime", "StepResult"]
