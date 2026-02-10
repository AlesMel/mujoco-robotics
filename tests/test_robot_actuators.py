"""Tests for reusable robot actuator profiles."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mujoco_robot.robots.actuators import (
    ROBOT_ACTUATOR_CONFIGS,
    configure_position_actuators,
    get_robot_actuator_config,
    resolve_robot_actuators,
)
from mujoco_robot.robots.configs import get_robot_config


def test_builtin_actuator_profiles_exist() -> None:
    """Built-in robots should have registered actuator profiles."""
    assert "ur3e" in ROBOT_ACTUATOR_CONFIGS
    assert "ur5e" in ROBOT_ACTUATOR_CONFIGS


def test_unknown_actuator_profile_raises() -> None:
    """Unknown actuator profiles should raise a clear error."""
    with pytest.raises(ValueError):
        get_robot_actuator_config("not-a-robot")


@pytest.mark.parametrize("robot_name", ["ur3e", "ur5e"])
def test_resolve_and_configure_position_actuators(robot_name: str) -> None:
    """Resolver should find all ids and apply control defaults consistently."""
    robot_cfg = get_robot_config(robot_name)
    model = mujoco.MjModel.from_xml_path(robot_cfg.model_path)

    handles = resolve_robot_actuators(model, robot_name)
    assert len(handles.joint_names) == 6
    assert len(handles.joint_ids) == 6
    assert len(handles.qpos_ids) == 6
    assert len(handles.dof_ids) == 6
    assert len(handles.actuator_ids) == 6

    for act_id in handles.actuator_ids:
        model.actuator_gainprm[act_id, 0] = 0.0

    configure_position_actuators(
        model,
        handles,
        min_damping=9.0,
        min_frictionloss=0.4,
        kp=350.0,
    )

    for dof_id in handles.dof_ids:
        assert model.dof_damping[dof_id] >= 9.0
        assert model.dof_frictionloss[dof_id] >= 0.4

    for joint_id, act_id in zip(handles.joint_ids, handles.actuator_ids):
        lo, hi = model.jnt_range[joint_id]
        expected = (
            np.array([lo, hi], dtype=float)
            if lo < hi
            else np.array(handles.config.fallback_ctrlrange, dtype=float)
        )
        np.testing.assert_allclose(model.actuator_ctrlrange[act_id], expected)
        assert model.actuator_gainprm[act_id, 0] == pytest.approx(350.0)
        if int(model.actuator_biastype[act_id]) == int(mujoco.mjtBias.mjBIAS_AFFINE):
            assert model.actuator_biasprm[act_id, 1] == pytest.approx(-350.0)
