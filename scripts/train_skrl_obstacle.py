#!/usr/bin/env python
"""Simple entrypoint for SKRL obstacle-avoidance training.

Usage::

    python scripts/train_skrl_obstacle.py
    python scripts/train_skrl_obstacle.py --cfg-name crazyflie_obstacle_skrl_stable
    python scripts/train_skrl_obstacle.py --total-timesteps 10000000 --n-envs 16 --device cuda:0
"""
from __future__ import annotations
from mujoco_robot.training.train_crazyflie_obstacle_skrl import main

if __name__ == "__main__":
    main()
