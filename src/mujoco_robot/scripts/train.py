    #!/usr/bin/env python
"""Unified training entry point.

Usage::

    # Train reach task
    python scripts/train.py --task reach --cfg-name ur5e_joint_pos --total-timesteps 500000

    # Train slot sorter
    python scripts/train.py --task slot_sorter --total-timesteps 1000000
"""
from __future__ import annotations

import argparse


def main():
    p = argparse.ArgumentParser(description="Train PPO on UR robot tasks.")
    p.add_argument(
        "--task", type=str, default="reach",
        choices=["reach", "slot_sorter", "cable_routing"],
        help="Which task to train (default: reach).",
    )
    p.add_argument(
        "--cfg-name",
        type=str,
        default="ur3e_joint_pos_dense_stable",
        help="Config profile name (used by reach/cable_routing tasks).",
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
            cfg_name=args.cfg_name,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            save_video=args.save_video,
            save_video_every=args.save_video_every,
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
    elif args.task == "cable_routing":
        from mujoco_robot.training.train_cable_routing import train_cable_routing_ppo
        train_cable_routing_ppo(
            cfg_name=args.cfg_name,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            save_video=args.save_video,
            save_video_every=args.save_video_every,
        )


if __name__ == "__main__":
    main()
