#!/usr/bin/env python
"""Unified training entry point.

Usage::

    # Train reach task
    python scripts/train.py --task reach --robot ur5e --total-timesteps 500000

    # Train slot sorter
    python scripts/train.py --task slot_sorter --total-timesteps 1000000
"""
from __future__ import annotations

import argparse


def main():
    p = argparse.ArgumentParser(description="Train PPO on UR robot tasks.")
    p.add_argument(
        "--task", type=str, default="reach",
        choices=["reach", "slot_sorter"],
        help="Which task to train (default: reach).",
    )
    p.add_argument(
        "--robot", type=str, default="ur3e",
        choices=["ur3e", "ur5e"],
        help="Robot model for reach task (default: ur3e).",
    )
    p.add_argument(
        "--control-variant", type=str, default="joint_pos",
        choices=["ik_rel", "ik_abs", "joint_pos", "joint_pos_isaac_reward"],
        help="Reach control variant (default: joint_pos).",
    )
    p.add_argument(
        "--reach-threshold", type=float, default=0.005,
        help="Reach success position threshold in metres (default: 0.03).",
    )
    p.add_argument(
        "--ori-threshold", type=float, default=0.05,
        help="Reach success orientation threshold in radians (default: 0.25).",
    )
    p.add_argument(
        "--progress-bar", action=argparse.BooleanOptionalAction, default=True,
        help="Use Stable-Baselines3 progress bar for reach training.",
    )
    p.add_argument(
        "--sb3-verbose", type=int, default=0, choices=[0, 1, 2],
        help="Stable-Baselines3 verbosity for reach training (default: 0).",
    )
    p.add_argument(
        "--success-hold-steps", type=int, default=10,
        help="Consecutive success steps required for hold-success in reach task.",
    )
    p.add_argument(
        "--success-bonus", type=float, default=0.25,
        help="One-time reward added when hold-success is first achieved.",
    )
    p.add_argument(
        "--stay-reward-weight", type=float, default=0.05,
        help="Per-second reward weight while hold-success is maintained.",
    )
    p.add_argument(
        "--resample-on-success", action=argparse.BooleanOptionalAction, default=False,
        help="Resample goal immediately after hold-success.",
    )
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-video-every", type=int, default=10_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    args = p.parse_args()

    if args.task == "reach":
        from mujoco_robot.training.train_reach import train_reach_ppo
        train_reach_ppo(
            robot=args.robot,
            control_variant=args.control_variant,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            save_video=args.save_video,
            save_video_every=args.save_video_every,
            reach_threshold=args.reach_threshold,
            ori_threshold=args.ori_threshold,
            success_hold_steps=args.success_hold_steps,
            success_bonus=args.success_bonus,
            stay_reward_weight=args.stay_reward_weight,
            resample_on_success=args.resample_on_success,
            progress_bar=args.progress_bar,
            sb3_verbose=args.sb3_verbose,
        )
    elif args.task == "slot_sorter":
        from mujoco_robot.training.train_slot_sorter import train_slot_sorter_ppo
        train_slot_sorter_ppo(
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            learning_rate=args.learning_rate,
            save_video=args.save_video,
            save_video_every=args.save_video_every,
        )


if __name__ == "__main__":
    main()
