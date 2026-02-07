#!/usr/bin/env python
"""Launch teleop for any UR robot environment.

Usage::

    # Reach task (keyboard)
    python scripts/teleop.py --task reach --robot ur5e

    # Slot sorter (keyboard)
    python scripts/teleop.py --task slot_sorter

    # Slot sorter (gamepad)
    python scripts/teleop.py --task slot_sorter --gamepad
"""
from __future__ import annotations

import argparse
import sys


def main():
    p = argparse.ArgumentParser(description="Interactive teleop for UR robots.")
    p.add_argument(
        "--task", type=str, default="reach",
        choices=["reach", "slot_sorter"],
        help="Which task to launch (default: reach).",
    )
    p.add_argument(
        "--robot", type=str, default="ur5e",
        choices=["ur5e", "ur3e"],
        help="Robot model for reach task (default: ur5e).",
    )
    p.add_argument(
        "--gamepad", action="store_true",
        help="Use DualShock/DualSense gamepad (slot_sorter only).",
    )
    args = p.parse_args()

    if args.task == "reach":
        from mujoco_robot.envs import URReachEnv
        from mujoco_robot.teleop import ReachTeleop

        env = URReachEnv(robot=args.robot, time_limit=0)
        print(f"Robot: {args.robot} | reach: {env._TOTAL_REACH:.2f} m")
        ReachTeleop(env).run()

    elif args.task == "slot_sorter":
        from mujoco_robot.envs import URSlotSorterEnv

        env = URSlotSorterEnv(time_limit=0)
        if args.gamepad:
            try:
                from mujoco_robot.teleop.gamepad import DualShockTeleop
                DualShockTeleop(env).run()
            except ImportError:
                print("pygame not installed; falling back to keyboard teleop.")
                from mujoco_robot.teleop import SlotSorterTeleop
                SlotSorterTeleop(env).run()
        else:
            from mujoco_robot.teleop import SlotSorterTeleop
            SlotSorterTeleop(env).run()


if __name__ == "__main__":
    main()
