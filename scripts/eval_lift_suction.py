#!/usr/bin/env python
"""Repo-root wrapper for interactive lift-suction evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure local `src/` imports work when running `python scripts/*.py`.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mujoco_robot.training.eval_lift_suction import main


if __name__ == "__main__":
    main()

