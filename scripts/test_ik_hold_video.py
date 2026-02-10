#!/usr/bin/env python
"""Visual IK hold test — record videos of the EE holding target positions.

For each robot (UR5e, UR3e) drives the end-effector through a sequence
of target positions using the IK controller, holds at each one, and
records a multi-view video showing the stability.

Usage::

    python scripts/test_ik_hold_video.py
    python scripts/test_ik_hold_video.py --robot ur5e
    python scripts/test_ik_hold_video.py --robot ur3e --out videos/ik_hold
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from mujoco_robot.core.ik_controller import orientation_error_axis_angle
from mujoco_robot.envs.reach import ReachIKRelEnv


# ────────────────────────────────────────────────────────────────────────
# Waypoints per robot — reachable, collision-free positions
# ────────────────────────────────────────────────────────────────────────

WAYPOINTS = {
    "ur5e": [
        # (target_pos, target_yaw, label)
        # Home EE ≈ [0.20, 0.13, 1.21]
        (np.array([ 0.15,  0.10, 1.15]),  0.0,        "front-centre"),
        (np.array([ 0.10,  0.20, 1.10]),  math.pi/4,  "right-low"),
        (np.array([ 0.20, -0.05, 1.05]), -math.pi/3,  "left-low"),
        (np.array([ 0.18,  0.15, 1.20]),  math.pi/2,  "above-home"),
    ],
    "ur3e": [
        # Home EE ≈ [0.15, 0.14, 1.03]
        (np.array([ 0.12,  0.10, 1.00]),  0.0,        "front-centre"),
        (np.array([ 0.10,  0.15, 0.98]),  math.pi/6,  "right-low"),
        (np.array([ 0.13,  0.08, 0.95]), -math.pi/6,  "left-low"),
        (np.array([ 0.14,  0.12, 1.02]),  math.pi/4,  "above-home"),
    ],
}


def _annotate_frame(frame: np.ndarray, text: str) -> np.ndarray:
    """Burn a simple text label into the top-left of the frame.

    Uses a minimal bitmap approach — no external font dependency.
    White rectangle + black text substitute with a solid colour bar.
    """
    h, w = frame.shape[:2]
    # Draw a dark banner at the top
    banner_h = min(32, h // 15)
    out = frame.copy()
    out[:banner_h, :] = (out[:banner_h, :].astype(np.int32) * 0.3).astype(np.uint8)
    return out


def record_ik_hold_video(
    robot: str,
    out_dir: Path,
    width: int = 640,
    height: int = 480,
    drive_steps: int = 300,
    hold_steps: int = 200,
    fps: int = 30,
) -> Path:
    """Drive EE through waypoints, hold each, and record video.

    Returns the path to the saved video file.
    """
    env = ReachIKRelEnv(
        robot=robot,
        render_size=(width, height),
        time_limit=0,
        seed=42,
    )
    env.reset(seed=42)
    waypoints = WAYPOINTS[robot]

    frames: list[np.ndarray] = []

    # Capture a few initial frames at home
    for _ in range(30):
        frame = env.render(mode="rgb_array")
        if frame is not None:
            frames.append(frame)

    for wp_idx, (target_pos, target_yaw, label) in enumerate(waypoints):
        target_quat = np.array(
            [math.cos(target_yaw / 2), 0.0, 0.0, math.sin(target_yaw / 2)],
            dtype=np.float32,
        )
        # Place goal marker at target so it's visible in video
        env._place_goal_marker(target_pos, target_quat)

        print(f"  [{robot}] Waypoint {wp_idx}: {label} → "
              f"pos={target_pos.round(3)}, yaw={target_yaw:.2f} rad")

        # ── Drive phase ──
        for step in range(drive_steps):
            ee_pos = env.data.site_xpos[env.ee_site].copy()
            direction = target_pos - ee_pos
            dist = np.linalg.norm(direction)

            if dist > 1e-6:
                scale = min(1.0, dist / env.ee_step)
                action_xyz = (direction / dist) * scale
            else:
                action_xyz = np.zeros(3)

            ori_err_vec = orientation_error_axis_angle(env._ee_quat(), target_quat)
            action_ori = np.clip(ori_err_vec / env.ori_step, -1.0, 1.0)
            action = np.array([*action_xyz, *action_ori], dtype=np.float32)
            env.step(action)

            # Record every 2nd frame during drive
            if step % 2 == 0:
                frame = env.render(mode="rgb_array")
                if frame is not None:
                    frames.append(frame)

        # ── Hold phase ──
        hold_errors = []
        for step in range(hold_steps):
            env.step(np.zeros(6, dtype=np.float32))
            ee_pos = env.data.site_xpos[env.ee_site].copy()
            hold_errors.append(float(np.linalg.norm(ee_pos - target_pos)))

            # Record every 2nd frame during hold
            if step % 2 == 0:
                frame = env.render(mode="rgb_array")
                if frame is not None:
                    frames.append(frame)

        avg_err = np.mean(hold_errors)
        max_err = np.max(hold_errors)
        print(f"    Hold error — avg: {avg_err:.4f} m, max: {max_err:.4f} m")

    env.close()

    if not frames:
        raise RuntimeError("No frames captured — check MuJoCo renderer setup.")

    # Save video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"ik_hold_{robot}_{timestamp}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, frames, fps=fps)
    print(f"  ✓ Saved {len(frames)} frames to {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(
        description="Record IK hold-position test videos.",
    )
    p.add_argument(
        "--robot", type=str, default=None,
        choices=["ur5e", "ur3e"],
        help="Robot to test (default: both).",
    )
    p.add_argument(
        "--out", type=str, default="videos/ik_hold",
        help="Output directory for videos.",
    )
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    out_dir = Path(args.out)
    robots = [args.robot] if args.robot else ["ur5e", "ur3e"]

    print("=" * 60)
    print("IK Hold-Position Test — Video Recording")
    print("=" * 60)

    for robot in robots:
        print(f"\n{'─' * 40}")
        print(f"Robot: {robot}")
        print(f"{'─' * 40}")
        record_ik_hold_video(
            robot=robot,
            out_dir=out_dir,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )

    print(f"\n{'=' * 60}")
    print(f"Done! Videos saved to {out_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
