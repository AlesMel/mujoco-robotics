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

from mujoco_robot.training.reach_cli import (
    add_reach_train_args,
    reach_train_kwargs_from_args,
)


def main():
    p = argparse.ArgumentParser(description="Train PPO on UR robot tasks.")
    p.add_argument(
        "--task", type=str, default="reach",
        choices=["reach", "slot_sorter"],
        help="Which task to train (default: reach).",
    )
    add_reach_train_args(
        p,
        default_total_timesteps=500_000,
        default_n_envs=8,
        control_variant_choices=None,
    )
    p.set_defaults(save_video_every=100_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    args = p.parse_args()

    if args.task == "reach":
        from mujoco_robot.training.train_reach import train_reach_ppo

        train_reach_ppo(**reach_train_kwargs_from_args(args))
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
