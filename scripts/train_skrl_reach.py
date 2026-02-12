#!/usr/bin/env python
"""Simple entrypoint for SKRL reach training.

Usage::

    python scripts/train_skrl_reach.py --robot ur3e --total-timesteps 24000
    python scripts/train_skrl_reach.py --robot ur5e --cfg-name ur5e_joint_pos
"""
from __future__ import annotations
from mujoco_robot.training.train_reach_skrl import main

if __name__ == "__main__":
    main()
