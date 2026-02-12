"""Robot configuration registry.

Each robot entry contains:
    model_path    — path to the MJCF XML (relative to package root)
    base_pos      — world position of the robot base
    link_lengths  — kinematic link lengths from UR ROS2 official description
    init_q        — collision-free home joint angles
    goal_bounds   — workspace volume for goal sampling [3 x 2]
    ee_bounds     — hard EE position clamps [3 x 2]
    goal_distance — (min, max) distance from base for goal sampling
    goal_min_height — minimum Z for goals (prevents folding onto table)
    goal_min_ee_dist — minimum distance from current EE (prevents trivial starts)

To add a new robot, create an MJCF in ``mujoco_robot/robots/`` and register
it here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Locate MJCF files relative to this module
# ---------------------------------------------------------------------------
_ROBOTS_DIR = Path(__file__).resolve().parent


@dataclass
class RobotConfig:
    """Immutable robot configuration."""

    name: str
    model_path: str
    base_pos: np.ndarray
    link_lengths: List[float]
    init_q: np.ndarray
    goal_bounds: np.ndarray   # (3, 2) — [axis, lo/hi]
    ee_bounds: np.ndarray     # (3, 2)
    goal_distance: Tuple[float, float] = (0.20, 0.90)  # (min, max) from base
    goal_min_height: float = 0.82   # min Z — above table surface
    goal_min_ee_dist: float = 0.15  # min distance from current EE

    @property
    def total_reach(self) -> float:
        return float(sum(self.link_lengths))


# ---------------------------------------------------------------------------
# Built-in robots
# ---------------------------------------------------------------------------
ROBOT_CONFIGS: Dict[str, RobotConfig] = {
    "ur5e": RobotConfig(
        name="ur5e",
        model_path=str(_ROBOTS_DIR / "ur5e.xml"),
        base_pos=np.array([-0.30, 0.0, 0.74]),
        # Official UR5e DH reach parameters: a2 + a3 + d5 + d6 ≈ 0.85 m
        link_lengths=[0.1625, 0.425, 0.3922, 0.1333, 0.0997, 0.0996],
        # Forward-facing home in this MJCF frame (toward table center).
        init_q=np.array([-0.2670, -np.pi / 2, np.pi / 2,
                         -np.pi / 2, -np.pi / 2, 0.0]),
        # Goal sampling volume (world frame, above table, IN FRONT of arm)
        # Base at x=−0.30, EE home at x≈+0.20 → goals span x∈[−0.10, 0.55]
        goal_bounds=np.array([
            [-0.10, 0.55],   # x — in front of base
            [-0.35, 0.35],   # y — lateral spread
            [ 0.85, 1.40],   # z — above table top (0.74)
        ]),
        ee_bounds=np.array([
            [-1.10, 0.60],
            [-0.80, 0.80],
            [ 0.50, 1.85],
        ]),
        # Goal must be 25–80 cm from base, ≥85 cm high, ≥20 cm from EE
        goal_distance=(0.25, 0.80),
        goal_min_height=0.85,
        goal_min_ee_dist=0.15,
    ),
    "ur3e": RobotConfig(
        name="ur3e",
        model_path=str(_ROBOTS_DIR / "ur3e.xml"),
        base_pos=np.array([-0.15, 0.0, 0.74]),
        # Official UR3e DH reach parameters: a2 + a3 + d5 + d6 ≈ 0.50 m
        link_lengths=[0.15185, 0.24355, 0.2132, 0.13105, 0.08535, 0.0921],
        # Forward-facing home in this MJCF frame (toward table center).
        init_q=np.array([-0.4147, -np.pi / 2, np.pi / 2,
                         -np.pi / 2, -np.pi / 2, 0.0]),
        # Goal sampling volume (world frame, above table, IN FRONT of arm)
        # Base at x=−0.15, EE home at x≈+0.15 → goals span x∈[−0.02, 0.35]
        goal_bounds=np.array([
            [-0.02, 0.35],   # x — in front of base
            [-0.20, 0.20],   # y — lateral spread
            [ 0.85, 1.10],   # z — above table top (0.74)
        ]),
        ee_bounds=np.array([
            [-0.65, 0.35],
            [-0.50, 0.50],
            [ 0.60, 1.45],
        ]),
        # Goal must be 15–45 cm from base, ≥85 cm high, ≥12 cm from EE
        goal_distance=(0.15, 0.45),
        goal_min_height=0.85,
        goal_min_ee_dist=0.12,
    ),
}


def get_robot_config(name: str) -> RobotConfig:
    """Look up a robot by name.  Raises ``ValueError`` for unknown names."""
    if name not in ROBOT_CONFIGS:
        raise ValueError(
            f"Unknown robot '{name}'.  Available: {list(ROBOT_CONFIGS)}"
        )
    return ROBOT_CONFIGS[name]
