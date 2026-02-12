"""Scene construction helpers for reach environments.

Extracts the MuJoCo model build, renderer creation, actuator setup,
collision detector, and IK controller init from ``URReachEnvBase.__init__``
into a reusable function.
"""
from __future__ import annotations

from typing import Any, Tuple

import mujoco
import numpy as np

from mujoco_robot.assets.actuators import (
    configure_position_actuators,
    resolve_robot_actuators,
)
from mujoco_robot.core.collision import CollisionDetector
from mujoco_robot.core.ik_controller import IKController
from mujoco_robot.core.xml_builder import load_robot_xml, build_reach_xml


def build_reach_scene(env: Any) -> None:
    """Build the MuJoCo scene and assign all artifacts onto *env*.

    Reads ``env.model_path``, ``env.render_size``, ``env.actuator_kp``,
    ``env.min_joint_damping``, ``env.min_joint_frictionloss``,
    ``env.ik_damping``, and ``env.robot`` to construct the scene.

    After this call, *env* will have:
    ``model``, ``data``, ``model_xml``, ``_renderer_top``,
    ``_renderer_side``, ``_renderer_ee``, ``ee_site``, ``goal_site``,
    ``goal_geom``, ``_base_body_id``, ``robot_joints``,
    ``robot_actuators``, ``robot_dofs``, ``_robot_joint_ids``,
    ``_robot_qpos_ids``, ``_collision_detector``, and ``_ik``.
    """
    # ---- XML & model ----
    robot_xml = load_robot_xml(env.model_path)
    marker_size = 0.06
    env.model_xml = build_reach_xml(robot_xml, env.render_size, marker_size)
    env.model = mujoco.MjModel.from_xml_string(env.model_xml)
    env.data = mujoco.MjData(env.model)

    # ---- Solver settings ----
    env.model.opt.timestep = 1.0 / 60.0
    env.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    env.model.opt.iterations = max(env.model.opt.iterations, 50)
    env.model.opt.gravity[:] = [0.0, 0.0, -9.81]

    # ---- Renderers ----
    rw, rh = env.render_size
    env._renderer_top = mujoco.Renderer(env.model, height=rh, width=rw)
    env._renderer_side = mujoco.Renderer(env.model, height=rh, width=rw)
    env._renderer_ee = mujoco.Renderer(env.model, height=rh, width=rw)

    # ---- Site / geom / body IDs ----
    env.ee_site = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
    )
    env.goal_site = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_SITE, "goal_site"
    )
    env.goal_geom = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_cube"
    )
    env._base_body_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_BODY, "base"
    )

    # ---- Table bounds & camera centering ----
    env._refresh_table_bounds()
    env._center_top_camera_over_table()

    # ---- Actuators ----
    actuator_handles = resolve_robot_actuators(env.model, env.robot)
    env.robot_joints = list(actuator_handles.joint_names)
    env.robot_actuators = list(actuator_handles.actuator_ids)
    env.robot_dofs = list(actuator_handles.dof_ids)
    env._robot_joint_ids = list(actuator_handles.joint_ids)
    env._robot_qpos_ids = list(actuator_handles.qpos_ids)

    configure_position_actuators(
        env.model,
        actuator_handles,
        min_damping=env.min_joint_damping,
        min_frictionloss=env.min_joint_frictionloss,
        kp=env.actuator_kp,
    )

    # ---- Self-collision detection ----
    env._collision_detector = CollisionDetector(env.model)
    env._self_collision_count = 0

    # ---- IK controller ----
    env._ik = IKController(
        env.model, env.data, env.ee_site, env.robot_dofs,
        damping=env.ik_damping,
    )
