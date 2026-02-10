"""Generic manager-based runtime scaffolding.

This package is a thin phase-1 skeleton to support IsaacLab-style task
packages. Task implementations can gradually migrate onto these base classes.
"""

from .base_env import ManagerBasedEnv
from .base_rl_env import ManagerBasedRLEnv
from .manager_runtime import ManagerRuntime

__all__ = [
    "ManagerBasedEnv",
    "ManagerBasedRLEnv",
    "ManagerRuntime",
]

