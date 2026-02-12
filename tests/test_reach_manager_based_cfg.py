"""Tests for manager-based reach cfg scaffolding."""

from __future__ import annotations

from mujoco_robot.tasks.reach.config import (
    get_reach_cfg,
    list_reach_cfgs,
)


def test_reach_manager_based_cfg_registry_contains_expected_profiles() -> None:
    names = list_reach_cfgs()
    assert "ur3e_joint_pos" in names
    assert "ur3e_ik_rel" in names
    assert "ur5e_joint_pos" in names
    assert "ur3e_joint_pos_dense_stable" in names


def test_reach_manager_based_cfg_to_legacy_kwargs_basic_mapping() -> None:
    cfg = get_reach_cfg("ur3e_joint_pos")
    kwargs = cfg.to_legacy_kwargs()

    assert cfg.scene.robot == "ur3e"
    assert cfg.actions.control_variant == "joint_pos"
    assert kwargs["joint_action_scale"] == cfg.actions.joint_action_scale
    assert kwargs["reach_threshold"] == cfg.success.reach_threshold
    assert kwargs["ori_threshold"] == cfg.success.ori_threshold
    assert "time_limit" not in kwargs

