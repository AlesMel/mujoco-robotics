"""Tests for shared reach training CLI wiring."""

from __future__ import annotations

import argparse
from mujoco_robot.training.reach_cli import (
    add_reach_train_args,
    reach_train_kwargs_from_args,
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_reach_train_args(p, control_variant_choices=None)
    return p


def test_reach_cli_supports_cfg_name() -> None:
    args = _parser().parse_args(["--cfg-name", "ur5e_joint_pos"])
    assert args.cfg_name == "ur5e_joint_pos"


def test_reach_cli_default_cfg_name_is_set() -> None:
    args = _parser().parse_args([])
    assert isinstance(args.cfg_name, str)
    assert args.cfg_name


def test_reach_cli_kwargs_mapping() -> None:
    args = _parser().parse_args(
        [
            "--robot",
            "ur5e",
            "--cfg-name",
            "ur5e_joint_pos",
            "--control-variant",
            "joint_pos",
            "--total-timesteps",
            "1234",
        ]
    )
    kwargs = reach_train_kwargs_from_args(args)
    assert kwargs["robot"] == "ur5e"
    assert kwargs["cfg_name"] == "ur5e_joint_pos"
    assert kwargs["control_variant"] == "joint_pos"
    assert kwargs["total_timesteps"] == 1234
