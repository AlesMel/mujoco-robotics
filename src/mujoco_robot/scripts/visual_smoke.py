#!/usr/bin/env python
"""Generate a short scripted rollout video for visual inspection.

Usage::

    python scripts/visual_smoke.py --out videos/smoke/smoke_test.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from mujoco_robot.envs import URSlotSorterEnv


def scripted_actions():
    """Gentle, collision-avoidant trajectory for visual QA."""
    seq = []
    seq += [np.array([0.0, 0.0, 0.7, 0.0, 0.0], dtype=np.float32)] * 8
    seq += [np.array([0.6, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)] * 8
    seq += [np.array([0.0, 0.6, 0.0, 0.0, 0.0], dtype=np.float32)] * 8
    seq += [np.array([0.0, 0.0, -0.4, 0.0, 0.0], dtype=np.float32)] * 6
    seq += [np.array([0.0, 0.0, 0.0, 0.4, 0.0], dtype=np.float32)] * 8
    seq += [np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)] * 4
    seq += [np.array([-0.6, -0.6, 0.5, 0.0, 1.0], dtype=np.float32)] * 10
    return seq


def render_rollout(out_path: Path, width: int = 480, height: int = 360) -> None:
    env = URSlotSorterEnv(render_size=(width, height), time_limit=0, seed=7)
    env.reset()
    frames = []
    for act in scripted_actions():
        step = env.step(act)
        frame = env.render(mode="rgb_array")
        if frame is not None:
            frames.append(frame)
        if step.done:
            break
    env.close()

    if not frames:
        raise RuntimeError("No frames captured; check MuJoCo headless/renderer setup.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, frames, fps=30)
    print(f"Saved {len(frames)} frames to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Render a short scripted rollout for visual QA.")
    p.add_argument("--out", type=Path, default=Path("videos/smoke/smoke_test.mp4"))
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=360)
    args = p.parse_args()
    render_rollout(args.out, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
