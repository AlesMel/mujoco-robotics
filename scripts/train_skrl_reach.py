#!/usr/bin/env python
"""Simple entrypoint for SKRL reach training.

Usage::

    python scripts/train_skrl_reach.py --robot ur3e --total-timesteps 24000
    python scripts/train_skrl_reach.py --robot ur5e --cfg-name ur5e_joint_pos
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure local `src/` imports work when running `python scripts/*.py`.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_PKG = _SRC / "mujoco_robot"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from training.train_reach_skrl import main


if __name__ == "__main__":
    main()
