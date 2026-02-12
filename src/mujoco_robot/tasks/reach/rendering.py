"""Rendering helpers for reach environments."""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _latest_metrics(env: Any) -> dict[str, float | bool]:
    """Return metric values from latest step info (fallback to live values)."""
    info = getattr(env, "_last_step_info", None)
    if isinstance(info, dict):
        dist = float(info.get("dist", env._ee_goal_dist()))
        ori_err_rad = float(info.get("ori_err", env._orientation_error_magnitude()))
        reach_threshold = float(info.get("reach_threshold", env.reach_threshold))
        ori_threshold = float(info.get("ori_threshold", env.ori_threshold))
        within_pos = bool(info.get("within_position_threshold", dist < reach_threshold))
        within_ori = bool(
            info.get("within_orientation_threshold", ori_err_rad < ori_threshold)
        )
    else:
        dist = float(env._ee_goal_dist())
        ori_err_rad = float(env._orientation_error_magnitude())
        reach_threshold = float(env.reach_threshold)
        ori_threshold = float(env.ori_threshold)
        within_pos = bool(dist < reach_threshold)
        within_ori = bool(ori_err_rad < ori_threshold)
    return {
        "dist": dist,
        "ori_err_rad": ori_err_rad,
        "reach_threshold": reach_threshold,
        "ori_threshold": ori_threshold,
        "within_pos": within_pos,
        "within_ori": within_ori,
    }


def _append_observation_lines(env: Any, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    """Append compact observation diagnostics from the last policy input."""
    obs = getattr(env, "_last_obs", None)
    if obs is None:
        return

    obs = np.asarray(obs, dtype=np.float32).ravel()
    n_j = int(len(getattr(env, "robot_joints", [])))
    action_dim = int(getattr(env, "action_dim", 0))

    lines.append(("", (255, 255, 255)))
    lines.append((f"Obs(raw) dim: {obs.shape[0]}", (180, 220, 255)))

    # Canonical reach observation layout: q_rel(6), qd(6), cmd(7), last_action(6).
    expected = (2 * n_j) + 7 + action_dim
    if n_j > 0 and action_dim > 0 and obs.shape[0] >= expected:
        idx = 0
        q_rel = obs[idx:idx + n_j]
        idx += n_j
        q_vel = obs[idx:idx + n_j]
        idx += n_j
        cmd = obs[idx:idx + 7]
        idx += 7
        last_action_obs = obs[idx:idx + action_dim]

        lines.append(
            (
                f"Obs q_rel/qd: {np.linalg.norm(q_rel):.3f} / {np.linalg.norm(q_vel):.3f}",
                (180, 220, 255),
            )
        )
        lines.append(
            (
                f"Obs cmd xyz:  {cmd[0]:+.3f} {cmd[1]:+.3f} {cmd[2]:+.3f}",
                (180, 220, 255),
            )
        )
        lines.append(
            (
                f"Obs cmd quat: {cmd[3]:+.3f} {cmd[4]:+.3f} {cmd[5]:+.3f} {cmd[6]:+.3f}",
                (180, 220, 255),
            )
        )
        lines.append(
            (
                f"Obs last act: {np.linalg.norm(last_action_obs):.3f}",
                (180, 220, 255),
            )
        )
        return

    # Fallback for non-canonical layouts: show first few values.
    preview_n = min(8, obs.shape[0])
    preview = " ".join(f"{v:+.3f}" for v in obs[:preview_n])
    lines.append((f"Obs[0:{preview_n}]: {preview}", (180, 220, 255)))


def draw_metrics_overlay(
    env: Any,
    frame: np.ndarray,
    panel_x: int | None = None,
    panel_y: int = 8,
    panel_w: int | None = None,
) -> np.ndarray:
    """Draw a translucent HUD with live metrics on a rendered frame."""
    h, w = frame.shape[:2]

    metrics = _latest_metrics(env)
    dist = float(metrics["dist"])
    ori_err_rad = float(metrics["ori_err_rad"])
    reach_threshold = float(metrics["reach_threshold"])
    ori_threshold = float(metrics["ori_threshold"])
    within_pos = bool(metrics["within_pos"])
    within_ori = bool(metrics["within_ori"])
    ori_err_deg = float(np.degrees(ori_err_rad))

    joint_pos_rad = np.array([env.data.qpos[qid] for qid in env._robot_qpos_ids])
    joint_vel_rad = env.data.qvel[env.robot_dofs]
    joint_pos_deg = np.degrees(joint_pos_rad)
    joint_vel_deg = np.degrees(joint_vel_rad)

    ee_pos = env.data.site_xpos[env.ee_site].copy()

    lines: list[tuple[str, tuple[int, int, int]]] = []
    lines.append((f"Step {env.step_id:>5d} / {env.time_limit}", (255, 255, 255)))
    lines.append((f"Reward:      {env._last_reward:+.4f}", (255, 220, 160)))
    lines.append(("", (255, 255, 255)))

    if within_pos:
        dist_color = (0, 255, 0)
    elif dist < reach_threshold * 5:
        dist_color = (255, 255, 0)
    else:
        dist_color = (255, 100, 100)
    lines.append((f"Distance:    {dist*100:6.2f} cm  ({dist:.4f} m)", dist_color))

    if within_ori:
        ori_color = (0, 255, 0)
    elif ori_err_rad < ori_threshold * 5:
        ori_color = (255, 255, 0)
    else:
        ori_color = (255, 100, 100)
    lines.append((f"Ori error:   {ori_err_deg:6.2f} deg ({ori_err_rad:.4f} rad)", ori_color))
    lines.append(
        (
            "Thresholds: "
            f"dist<{reach_threshold:.4f} m "
            f"ori<{np.degrees(ori_threshold):.2f} deg",
            (170, 170, 170),
        )
    )
    lines.append(("", (255, 255, 255)))

    lines.append(
        (
            f"EE pos:  x={ee_pos[0]:+.3f}  y={ee_pos[1]:+.3f}  z={ee_pos[2]:+.3f} m",
            (200, 200, 255),
        )
    )
    lines.append(
        (
            f"Goal:    x={env.goal_pos[0]:+.3f}  y={env.goal_pos[1]:+.3f}  z={env.goal_pos[2]:+.3f} m",
            (200, 255, 200),
        )
    )
    lines.append(("", (255, 255, 255)))

    lines.append(("Joint        Angle (deg)  Vel (deg/s)", (180, 180, 180)))
    lines.append(("-" * 40, (100, 100, 100)))
    for j in range(len(env.robot_joints)):
        jname = env.robot_joints[j]
        short = jname[-12:] if len(jname) > 12 else jname
        lines.append(
            (
                f"{short:<12s} {joint_pos_deg[j]:+8.2f}     {joint_vel_deg[j]:+8.2f}",
                (220, 220, 220),
            )
        )
    lines.append(("", (255, 255, 255)))

    lines.append((f"Goals reached:    {env._success_tracker.goals_reached}", (180, 220, 255)))
    lines.append((f"Goals held:       {env._success_tracker.goals_held}", (180, 220, 255)))
    lines.append((f"Goals resampled:  {env._goals_resampled}", (180, 220, 255)))
    lines.append((f"Self-collisions:  {env._self_collision_count}", (180, 220, 255)))
    _append_observation_lines(env, lines)

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSansMono.ttf", 13)
        except (IOError, OSError):
            font = ImageFont.load_default()

    line_h = 16
    pad = 6
    if panel_w is None:
        panel_w = 340
    if panel_x is None:
        panel_x = w - panel_w - 8
    panel_x = int(max(0, panel_x))
    panel_y = int(max(0, panel_y))
    max_panel_w = max(40, w - panel_x - 8)
    panel_w = int(max(40, min(panel_w, max_panel_w)))
    max_panel_h = max(40, h - panel_y - 8)

    def _max_lines_for_height(height_px: int) -> int:
        return max(1, (max_panel_h - 2 * pad) // height_px)

    max_lines = _max_lines_for_height(line_h)
    # If content is slightly too tall, tighten line spacing first so we keep
    # all rows visible instead of truncating the latest metrics.
    if len(lines) > max_lines:
        for compact_h in (15, 14, 13):
            compact_max = _max_lines_for_height(compact_h)
            if len(lines) <= compact_max:
                line_h = compact_h
                max_lines = compact_max
                break
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [("...", (180, 180, 180))]
    panel_h = pad * 2 + line_h * len(lines)

    draw.rectangle(
        [panel_x, panel_y, panel_x + panel_w, panel_y + panel_h],
        fill=(0, 0, 0, 160),
    )

    y = panel_y + pad
    for text, color in lines:
        if text:
            draw.text((panel_x + pad, y), text, fill=(*color, 255), font=font)
        y += line_h

    return np.array(img)


def compose_multi_camera_frame(env: Any) -> np.ndarray:
    """Render top/side/ee cameras and compose the 2x2 evaluation grid."""
    env._renderer_top.update_scene(env.data, camera="top")
    frame_top = env._renderer_top.render()
    env._renderer_side.update_scene(env.data, camera="side")
    frame_side = env._renderer_side.render()
    env._renderer_ee.update_scene(env.data, camera="ee_cam")
    frame_ee = env._renderer_ee.render()
    top_row = np.concatenate([frame_top, frame_side], axis=1)
    stats_frame = np.full_like(frame_ee, 18)
    stats_frame = env._draw_metrics_overlay(
        stats_frame,
        panel_x=8,
        panel_y=8,
        panel_w=stats_frame.shape[1] - 16,
    )
    bottom_row = np.concatenate([frame_ee, stats_frame], axis=1)
    return np.concatenate([top_row, bottom_row], axis=0)
