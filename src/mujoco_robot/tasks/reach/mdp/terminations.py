"""Termination terms for the reach task."""
from __future__ import annotations


def success(env, ctx: dict[str, float]) -> bool:
    return bool(ctx["dist"] < env.reach_threshold and ctx["ori_err"] < env.ori_threshold)


def failure_self_collision(env, _ctx: dict[str, float]) -> bool:
    return bool(env._self_collision_count > 0)


def time_out(env, _ctx: dict[str, float]) -> bool:
    return bool(env.time_limit > 0 and (env.step_id + 1) >= env.time_limit)
