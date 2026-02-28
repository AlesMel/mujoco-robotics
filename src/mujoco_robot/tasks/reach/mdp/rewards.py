"""Reward terms and weight helpers for the reach task."""
from __future__ import annotations

import math


def _safe_std(std: float) -> float:
    return max(1e-6, float(std))


def position_command_error(_env, ctx: dict[str, float]) -> float:
    return float(ctx["dist"])


def position_command_error_tanh(_env, ctx: dict[str, float], std: float = 0.1) -> float:
    return float(1.0 - math.tanh(ctx["dist"] / std))


def position_command_error_exp(_env, ctx: dict[str, float], std: float = 0.05) -> float:
    x = float(ctx["dist"]) / _safe_std(std)
    return float(math.exp(-(x * x)))


def orientation_command_error(_env, ctx: dict[str, float]) -> float:
    return float(ctx["ori_err"])


def orientation_error_tanh(_env, ctx: dict[str, float], std: float = 0.2) -> float:
    return float(1.0 - math.tanh(ctx["ori_err"] / std))


def orientation_command_error_exp(_env, ctx: dict[str, float], std: float = 0.2) -> float:
    x = float(ctx["ori_err"]) / _safe_std(std)
    return float(math.exp(-(x * x)))


def position_command_error_tanh_std_01(env, ctx: dict[str, float]) -> float:
    return position_command_error_tanh(env, ctx, std=0.1)


def position_command_error_tanh_with_std(std: float):
    """Return a position tanh reward function with custom std."""

    def _fn(env, ctx: dict[str, float]) -> float:
        return position_command_error_tanh(env, ctx, std=float(std))

    return _fn


def position_command_error_exp_with_std(std: float):
    """Return a position exp reward function with custom std."""

    def _fn(env, ctx: dict[str, float]) -> float:
        return position_command_error_exp(env, ctx, std=float(std))

    return _fn


def orientation_error_tanh_with_std(std: float):
    """Return an orientation tanh reward function with custom std."""

    def _fn(env, ctx: dict[str, float]) -> float:
        return orientation_error_tanh(env, ctx, std=float(std))

    return _fn


def orientation_command_error_exp_with_std(std: float):
    """Return an orientation exp reward function with custom std."""

    def _fn(env, ctx: dict[str, float]) -> float:
        return orientation_command_error_exp(env, ctx, std=float(std))

    return _fn


def orientation_error_linear(_env, ctx: dict[str, float]) -> float:
    """Linear orientation reward: 1 - err/π.  Constant gradient everywhere."""
    return float(1.0 - min(ctx["ori_err"], math.pi) / math.pi)


def orientation_error_tanh_std_02(env, ctx: dict[str, float]) -> float:
    return orientation_error_tanh(env, ctx, std=0.2)


def at_goal(env, ctx: dict[str, float]) -> float:
    return float(
        ctx["dist"] < env.reach_threshold and ctx["ori_err"] < env.ori_threshold
    )


def action_rate_l2(_env, ctx: dict[str, float]) -> float:
    return float(ctx["action_rate_l2"])


def joint_vel_l2(_env, ctx: dict[str, float]) -> float:
    return float(ctx["joint_vel_l2"])


def action_rate_curriculum_weight(env) -> float:
    return -float(env._curriculum_weight(env.action_rate_weight, env._action_rate_target))


def joint_vel_curriculum_weight(env) -> float:
    return -float(env._curriculum_weight(env.joint_vel_weight, env._joint_vel_target))


def scaled_by_step_dt(base_weight: float):
    """Return a callable weight that scales by environment step duration."""
    def _weight(env) -> float:
        return float(base_weight * env.model.opt.timestep * env.n_substeps)

    return _weight
