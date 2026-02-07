"""Core engine modules — IK, collision detection, XML utilities."""

from mujoco_robot.core.ik_controller import IKController
from mujoco_robot.core.collision import CollisionDetector
from mujoco_robot.core.xml_builder import load_robot_xml, inject_goal_marker, inject_side_camera

__all__ = [
    "IKController",
    "CollisionDetector",
    "load_robot_xml",
    "inject_goal_marker",
    "inject_side_camera",
]
