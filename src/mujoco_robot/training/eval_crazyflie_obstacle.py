#!/usr/bin/env python
"""Interactive evaluation for a trained Crazyflie obstacle-avoidance policy.

Renders the 4-camera composite view (with lidar rays) in an **OpenCV window**
with reliable keyboard input.  The trained PPO policy runs in real-time while
the user moves the green goal marker with the keyboard or mouse.

Controls  (click the OpenCV window first to give it focus!)
--------
    W / ↑        : goal forward  (+X)
    S / ↓        : goal backward (−X)
    A / ←        : goal left     (+Y)
    D / →        : goal right    (−Y)
    E            : goal up       (+Z)
    Q            : goal down     (−Z)
    B            : sample goal behind a barrier (maze-aware)
    Space        : reset episode
    P            : pause / resume
    T            : toggle auto-orbit (goal circles automatically)
    F            : snap goal to drone position
    R            : random new goal
    M            : toggle mouse-follow mode (goal tracks cursor over top-down view)
    Left-click   : place goal at clicked position (top-down view)
    Scroll wheel : adjust goal height (Z) in mouse mode
    +  /  =      : increase goal speed
    −            : decrease goal speed
    Escape       : quit

Usage::

    python -m mujoco_robot.training.eval_crazyflie_obstacle \\
        --model ppo_crazyflie_obstacle_crazyflie_obstacle_dense_stable.zip

    python scripts/eval_crazyflie_obstacle.py \\
        --model runs/crazyflie_obstacle_ppo_18/ppo_crazyflie_obstacle_crazyflie_obstacle_dense_stable.zip
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mujoco_robot.tasks import (
    get_crazyflie_cfg,
    make_crazyflie_obstacle_gymnasium,
)


# ──────────────────────────────────────────────────────────
# OpenCV key codes  (cv2.waitKeyEx, platform-dependent)
# ──────────────────────────────────────────────────────────
_K_ESC = 27
_K_SPACE = 32
_K_PLUS = ord("+")
_K_EQUAL = ord("=")
_K_MINUS = ord("-")

# Arrow keys — Linux X11 via waitKeyEx
_K_UP = 0xFF52
_K_DOWN = 0xFF54
_K_LEFT = 0xFF51
_K_RIGHT = 0xFF53
# Fallback (some systems)
_K_UP2 = 0x00520000
_K_DOWN2 = 0x00540000
_K_LEFT2 = 0x00510000
_K_RIGHT2 = 0x00530000


def _is_key(code: int, *targets: int) -> bool:
    return code in targets


# ──────────────────────────────────────────────────────────
# Mouse ↔ world projection for the top-down (cf_top) camera
# ──────────────────────────────────────────────────────────
def _pixel_to_world_xy(
    px: int,
    py: int,
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    cam_name: str,
    render_w: int,
    render_h: int,
    target_z: float,
) -> tuple[float, float] | None:
    """Project a pixel in a single camera tile to world XY at *target_z*.

    Uses the camera intrinsics / extrinsics from the *live* MjData so that
    trackcom cameras work correctly.

    Returns ``(world_x, world_y)`` or ``None`` when the ray is nearly
    parallel to the target plane.
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        return None

    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    fovy = math.radians(float(model.cam_fovy[cam_id]))
    aspect = render_w / render_h

    half_h = math.tan(fovy / 2.0)
    half_w = half_h * aspect

    # Normalised device coords – image (0,0) is top-left
    ndc_x = 2.0 * px / render_w - 1.0
    ndc_y = 1.0 - 2.0 * py / render_h

    dir_cam = np.array([ndc_x * half_w, ndc_y * half_h, -1.0])
    dir_cam /= np.linalg.norm(dir_cam)
    dir_world = cam_mat @ dir_cam

    # Intersect with horizontal plane z = target_z
    if abs(dir_world[2]) < 1e-8:
        return None
    t = (target_z - cam_pos[2]) / dir_world[2]
    if t < 0:
        return None
    hit = cam_pos + t * dir_world
    return float(hit[0]), float(hit[1])


class _MouseState:
    """Shared mutable state written by the OpenCV mouse callback."""

    def __init__(self) -> None:
        self.follow_mode: bool = False   # M toggles continuous tracking
        self.last_px: int = -1           # pixel x inside the cf_top tile
        self.last_py: int = -1           # pixel y inside the cf_top tile
        self.click_pending: bool = False # set True on left-click
        self.scroll_delta: int = 0       # accumulated scroll ticks
        self.inside_top: bool = False    # cursor is over the top-down panel


def _make_mouse_callback(
    mouse: _MouseState,
    tile_w: int,
    tile_h: int,
    display_scale: float,
):
    """Return an OpenCV mouse callback that populates *mouse*.

    The top-down camera occupies the **top-right** tile of the 2×2 composite.
    """
    # In the displayed (possibly scaled) image:
    x_min = int(tile_w * display_scale)
    x_max = int(2 * tile_w * display_scale)
    y_min = 0
    y_max = int(tile_h * display_scale)

    def _cb(event: int, x: int, y: int, flags: int, param) -> None:
        if x_min <= x < x_max and y_min <= y < y_max:
            mouse.inside_top = True
            # Map back to the tile's own pixel coords (un-scaled)
            mouse.last_px = int((x - x_min) / display_scale)
            mouse.last_py = int((y - y_min) / display_scale)

            if event == cv2.EVENT_LBUTTONDOWN:
                mouse.click_pending = True

            if event == cv2.EVENT_MOUSEWHEEL:
                # flags encodes scroll direction
                mouse.scroll_delta += 1 if flags > 0 else -1
        else:
            mouse.inside_top = False

    return _cb


# ──────────────────────────────────────────────────────────
# Model / VecNorm utilities
# ──────────────────────────────────────────────────────────
def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    if p.suffix != ".zip":
        z = p.with_suffix(".zip")
        if z.exists():
            return z
    raise FileNotFoundError(f"Not found: '{path_str}' (tried .zip too)")


def _default_vecnorm_path(model_path: Path) -> Path:
    stem = model_path.with_suffix("") if model_path.suffix == ".zip" else model_path
    return Path(f"{stem}_vecnorm.pkl")


def _load_vec_env(env, vecnorm_path: Path) -> VecNormalize:
    vec_env = DummyVecEnv([lambda: env])
    if not vecnorm_path.exists():
        raise FileNotFoundError(
            f"VecNormalize stats not found: '{vecnorm_path}'. "
            "Ensure <model>_vecnorm.pkl exists next to the model."
        )
    vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    print(f"[eval] Loaded VecNormalize stats: {vecnorm_path}")
    return vec_env


# ──────────────────────────────────────────────────────────
# HUD overlay  (cv2.putText — with obstacle-specific telemetry)
# ──────────────────────────────────────────────────────────
def _draw_hud(
    frame: np.ndarray,
    goal: np.ndarray,
    ep_steps: int,
    ep_return: float,
    goal_speed: float,
    paused: bool,
    auto_orbit: bool,
    ep_done: bool,
    info: dict,
) -> np.ndarray:
    """Burn a translucent HUD bar onto the top and bottom of the frame."""
    h, w = frame.shape[:2]

    # ---- Top banner (semi-transparent) ----
    bar_h = 74
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.48
    c_white = (255, 255, 255)
    c_green = (100, 255, 120)
    c_yellow = (80, 230, 255)
    c_red = (80, 80, 255)
    c_orange = (60, 180, 255)
    c_cyan = (255, 220, 100)

    dist = float(info.get("goal_dist", info.get("pos_error", 0.0)))
    goals_reached = int(info.get("goals_reached", 0))
    tilt = float(info.get("tilt_deg", 0.0))

    # Row 1 — goal position, distance, tilt, goals
    txt1 = (
        f"Goal: [{goal[0]:+.2f}, {goal[1]:+.2f}, {goal[2]:+.2f}]   "
        f"Dist: {dist:.3f}m   Tilt: {tilt:.1f}deg   Goals: {goals_reached}"
    )
    cv2.putText(frame, txt1, (10, 18), font, fs, c_green, 1, cv2.LINE_AA)

    # Row 2 — step, return, speed, state
    state = "PAUSED" if paused else ("DONE - Space to reset" if ep_done else "RUNNING")
    state_color = c_yellow if paused else (c_red if ep_done else c_white)
    orbit_str = "  [ORBIT]" if auto_orbit else ""
    txt2 = (
        f"Step: {ep_steps:5d}  Return: {ep_return:+.1f}  "
        f"Speed: {goal_speed:.1f}m/s{orbit_str}   |  {state}"
    )
    cv2.putText(frame, txt2, (10, 38), font, fs, state_color, 1, cv2.LINE_AA)

    # Row 3 — obstacle-specific telemetry
    n_walls = int(info.get("n_active_walls", info.get("n_active_obstacles", 0)))
    min_range = float(info.get("min_obstacle_range_m", -1.0))
    spd_match = float(info.get("speed_match", info.get("R_direction", 0.0)))
    stab_fac = float(
        info.get(
            "stability_factor",
            np.clip(1.0 + float(info.get("R_stability", -1.0)), 0.0, 1.0),
        )
    )
    vel_toward = float(info.get("vel_toward_goal", 0.0))
    prox_pen = float(info.get("obstacle_proximity_penalty", 0.0))
    collision = bool(info.get("obstacle_collision", False))

    range_str = f"{min_range:.2f}m" if min_range >= 0.0 else "n/a"
    txt3 = (
        f"Walls: {n_walls}  MinRange: {range_str}  "
        f"Spd: {spd_match:.2f}  Stab: {stab_fac:.2f}  V->g: {vel_toward:+.2f}  Prox: {prox_pen:.2f}"
    )
    cv2.putText(frame, txt3, (10, 60), font, fs, c_cyan, 1, cv2.LINE_AA)

    # Collision flash
    if collision:
        cv2.putText(
            frame, "!! COLLISION !!", (w - 220, 60),
            font, 0.65, c_red, 2, cv2.LINE_AA,
        )

    # ---- Bottom hint bar ----
    hint_h = 22
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - hint_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)
    mouse_tag = "  [MOUSE]" if info.get("_mouse_follow") else ""
    hint = (
        "WASD:move  Q/E:up/dn  B:barrier-goal  M:mouse  "
        "Click:place  Scroll:Z  T:orbit  F:snap  R:random  P:pause  Esc:quit"
        + mouse_tag
    )
    cv2.putText(frame, hint, (10, h - 6), font, 0.36, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


# ──────────────────────────────────────────────────────────
# NN input overlay  (translucent panel on the right side)
# ──────────────────────────────────────────────────────────

# Observation layout for state_estimate mode (27 parent + N lidar)
_OBS_SEGMENTS = [
    ("pos_err_body",  3),
    ("vel_body",      3),
    ("quat",          4),
    ("gyro",          3),
    ("accel",         3),
    ("baro_alt",      1),
    ("motor_norm",    4),
    ("battery",       1),
    ("last_action",   4),
    ("goals_norm",    1),
    # rangefinder filled dynamically
]


def _draw_nn_inputs(
    frame: np.ndarray,
    raw_obs: np.ndarray | None,
    norm_obs: np.ndarray | None,
    n_lidar: int,
) -> np.ndarray:
    """Draw a translucent panel on the right side showing NN input values."""
    if raw_obs is None:
        return frame
    h, w = frame.shape[:2]

    # Panel dimensions
    panel_w = 310
    x0 = w - panel_w
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.34
    line_h = 14
    c_label = (180, 180, 180)
    c_value = (255, 255, 255)
    c_header = (100, 255, 120)
    c_lidar = (255, 220, 100)

    # Build segment list (including dynamic lidar count)
    segments = list(_OBS_SEGMENTS)
    lidar_total = len(raw_obs) - sum(s[1] for s in segments)
    if lidar_total > 0:
        segments.append((f"lidar({lidar_total})", lidar_total))

    # Count lines needed
    n_lines = 2  # header lines
    for name, dim in segments:
        n_lines += 1  # segment header
        if dim <= 4:
            n_lines += 1
        else:
            n_lines += (dim + 7) // 8  # 8 values per row for lidar
    n_lines += 2  # padding

    panel_h = min(n_lines * line_h + 10, h - 80)
    y0 = 78  # below the top HUD bar

    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (w, y0 + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    y = y0 + 16
    cv2.putText(frame, "NN INPUTS (raw)", (x0 + 6, y), font, 0.42, c_header, 1, cv2.LINE_AA)
    y += line_h + 2

    idx = 0
    for name, dim in segments:
        if y > y0 + panel_h - 10:
            break
        # Segment label
        cv2.putText(frame, f"{name}:", (x0 + 6, y), font, fs, c_label, 1, cv2.LINE_AA)
        y += line_h

        if idx + dim <= len(raw_obs):
            vals = raw_obs[idx: idx + dim]
        else:
            vals = np.zeros(dim)

        if dim <= 4:
            txt = "  ".join(f"{v:+.3f}" for v in vals)
            cv2.putText(frame, txt, (x0 + 10, y), font, fs, c_value, 1, cv2.LINE_AA)
            y += line_h
        else:
            # Compact rows of 8 for lidar
            color = c_lidar
            for row_start in range(0, dim, 8):
                if y > y0 + panel_h - 10:
                    break
                chunk = vals[row_start: row_start + 8]
                txt = " ".join(f"{v:.2f}" for v in chunk)
                cv2.putText(frame, txt, (x0 + 10, y), font, fs, color, 1, cv2.LINE_AA)
                y += line_h

        idx += dim

    return frame


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive Crazyflie obstacle-avoidance eval – OpenCV window, keyboard controls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model", type=str, required=True,
        help="Path to SB3 PPO model (.zip).",
    )
    p.add_argument(
        "--vecnorm", type=str, default=None,
        help="VecNormalize .pkl. If omitted, <model>_vecnorm.pkl is used.",
    )
    p.add_argument(
        "--cfg-name", type=str, default="crazyflie_obstacle_dense_stable",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True,
        help="Use deterministic policy actions.",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--time-limit", type=int, default=0,
        help="Episode length. 0 = unlimited (manual reset only).",
    )
    p.add_argument(
        "--goal-speed", type=float, default=0.4,
        help="Keyboard goal movement speed (m/s).",
    )
    p.add_argument(
        "--print-every", type=int, default=60,
        help="Print telemetry every N control steps. 0 = off.",
    )
    p.add_argument(
        "--window-scale", type=float, default=1.0,
        help="Scale the display window (e.g. 1.5 for 150%%).",
    )
    return p


# ──────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────
_WINDOW_NAME = "Crazyflie Obstacle - Interactive Eval"


def main() -> None:
    args = _parser().parse_args()

    model_path = _resolve_path(args.model)
    vecnorm_path = (
        Path(args.vecnorm) if args.vecnorm else _default_vecnorm_path(model_path)
    )

    # ---- Build environment with rgb_array rendering ----
    cfg = get_crazyflie_cfg(args.cfg_name)
    cfg.seed = args.seed
    cfg.render = True
    cfg.render_mode = "rgb_array"
    if args.time_limit > 0:
        cfg.time_limit = args.time_limit
    else:
        cfg.time_limit = 100_000  # effectively unlimited

    gym_env = make_crazyflie_obstacle_gymnasium(cfg)
    vec_env = _load_vec_env(Monitor(gym_env), vecnorm_path)
    model = PPO.load(str(model_path), env=vec_env, device=args.device)

    # Direct access to the raw CrazyflieObstacleEnv
    base = gym_env.base
    ctrl_dt = float(base.model.opt.timestep) * float(base.n_substeps)

    # Workspace bounds
    xy_lim = base.workspace_xy
    z_lo, z_hi = base.workspace_z

    # ---- State ----
    goal_speed = args.goal_speed
    paused = False
    auto_orbit = False
    orbit_speed = 0.6
    orbit_phase = 0.0
    ep_return = 0.0
    ep_steps = 0
    n_episodes = 0
    ep_done = False
    last_info: dict = {}
    last_raw_obs: np.ndarray | None = None
    last_norm_obs: np.ndarray | None = None

    # ---- Print help ----
    print("\n" + "=" * 66)
    print("  Crazyflie Obstacle – Interactive Evaluation  (OpenCV window)")
    print("=" * 66)
    print("  WASD / Arrows  : move goal in XY")
    print("  E / Q          : goal up / down")
    print("  + / -          : faster / slower goal movement")
    print("  B              : sample goal behind a barrier (maze-aware)")
    print("  Space          : reset episode")
    print("  P              : pause / resume")
    print("  T              : toggle auto-orbit")
    print("  F              : snap goal to drone position")
    print("  R              : random new goal (free placement)")
    print("  M              : toggle mouse-follow mode")
    print("  Left-click     : place goal (top-down view)")
    print("  Scroll wheel   : adjust goal Z in mouse mode")
    print("  Escape         : quit")
    print()
    print("  Lidar rays are drawn in the 3D views (green=far, red=close).")
    print("  Walls are shown on the minimap (red/gold/purple).")
    print()
    print("  >>> Click the OpenCV window to give it keyboard focus <<<")
    print("=" * 66 + "\n")

    # ---- Initial reset ----
    obs = vec_env.reset()

    # ---- OpenCV window setup ----
    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    init_frame = base.render(mode="rgb_array")
    fh, fw = init_frame.shape[:2]
    disp_w = int(fw * args.window_scale)
    disp_h = int(fh * args.window_scale)
    cv2.resizeWindow(_WINDOW_NAME, disp_w, disp_h)

    # ---- Mouse interaction ----
    tile_w, tile_h = base.render_size   # single camera tile size
    mouse = _MouseState()
    cv2.setMouseCallback(
        _WINDOW_NAME,
        _make_mouse_callback(mouse, tile_w, tile_h, args.window_scale),
    )

    running = True
    last_tick = time.perf_counter()
    need_step = True  # whether enough time has elapsed for a physics step

    while running:
        now = time.perf_counter()
        dt_real = now - last_tick
        need_step = dt_real >= ctrl_dt

        # ── Keyboard input (1ms poll — keeps the window responsive) ──
        raw_key = cv2.waitKeyEx(1)
        key = raw_key & 0xFFFFFF

        # --- One-shot actions ---
        if key == _K_ESC:
            running = False
            break

        if _is_key(key, _K_SPACE):
            obs = vec_env.reset()
            ep_return = 0.0
            ep_steps = 0
            ep_done = False
            paused = False
            last_info = {}
            print("[eval] Episode reset.")

        elif _is_key(key, ord("p"), ord("P")):
            paused = not paused
            print(f"[eval] {'Paused' if paused else 'Resumed'}")

        elif _is_key(key, ord("t"), ord("T")):
            auto_orbit = not auto_orbit
            orbit_phase = math.atan2(base._goal_pos[1], base._goal_pos[0])
            print(f"[eval] Auto-orbit: {'ON' if auto_orbit else 'OFF'}")

        elif _is_key(key, ord("f"), ord("F")):
            pos = base.data.qpos[base._cf_qpos_adr: base._cf_qpos_adr + 3].copy()
            base._goal_pos[:] = pos
            base._update_goal_marker()
            base._reach_hold_counter = 0
            print(f"[eval] Goal snapped to drone: "
                  f"[{pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}]")

        elif _is_key(key, ord("b"), ord("B")):
            # Sample goal behind a barrier using the maze-aware method
            base._goal_pos[:] = base._sample_goal()
            base._update_goal_marker()
            base._reach_hold_counter = 0
            print(f"[eval] Barrier goal: "
                  f"[{base._goal_pos[0]:+.2f},{base._goal_pos[1]:+.2f},"
                  f"{base._goal_pos[2]:+.2f}]")

        elif _is_key(key, ord("r"), ord("R")):
            # Free random goal (parent method, ignores barriers)
            gp = np.array([
                base._rng.uniform(-xy_lim, xy_lim),
                base._rng.uniform(-xy_lim, xy_lim),
                base._rng.uniform(z_lo, z_hi),
            ])
            base._goal_pos[:] = gp
            base._update_goal_marker()
            base._reach_hold_counter = 0
            print(f"[eval] Random goal: "
                  f"[{base._goal_pos[0]:+.2f},{base._goal_pos[1]:+.2f},"
                  f"{base._goal_pos[2]:+.2f}]")

        elif _is_key(key, ord("m"), ord("M")):
            mouse.follow_mode = not mouse.follow_mode
            print(f"[eval] Mouse-follow: {'ON – hover over top-down view' if mouse.follow_mode else 'OFF'}")

        elif _is_key(key, _K_PLUS, _K_EQUAL):
            goal_speed = min(goal_speed + 0.1, 2.0)
            print(f"[eval] Goal speed: {goal_speed:.1f} m/s")

        elif _is_key(key, _K_MINUS):
            goal_speed = max(goal_speed - 0.1, 0.1)
            print(f"[eval] Goal speed: {goal_speed:.1f} m/s")

        # --- Continuous movement keys ---
        goal_delta = np.zeros(3, dtype=float)
        if _is_key(key, ord("w"), ord("W"), _K_UP, _K_UP2):
            goal_delta[0] += goal_speed
        if _is_key(key, ord("s"), ord("S"), _K_DOWN, _K_DOWN2):
            goal_delta[0] -= goal_speed
        if _is_key(key, ord("a"), ord("A"), _K_LEFT, _K_LEFT2):
            goal_delta[1] += goal_speed
        if _is_key(key, ord("d"), ord("D"), _K_RIGHT, _K_RIGHT2):
            goal_delta[1] -= goal_speed
        if _is_key(key, ord("e"), ord("E")):
            goal_delta[2] += goal_speed
        if _is_key(key, ord("q"), ord("Q")):
            goal_delta[2] -= goal_speed

        if np.any(np.abs(goal_delta) > 1e-6):
            # Use real dt so holding a key gives smooth movement
            dt_move = max(dt_real, ctrl_dt)
            base._goal_pos += goal_delta * dt_move
            base._goal_pos[0] = float(np.clip(base._goal_pos[0], -xy_lim, xy_lim))
            base._goal_pos[1] = float(np.clip(base._goal_pos[1], -xy_lim, xy_lim))
            base._goal_pos[2] = float(np.clip(base._goal_pos[2], z_lo, z_hi))
            base._update_goal_marker()

        # --- Mouse goal control ---
        _z_step = 0.03  # metres per scroll tick
        if mouse.scroll_delta != 0:
            base._goal_pos[2] += mouse.scroll_delta * _z_step
            base._goal_pos[2] = float(np.clip(base._goal_pos[2], z_lo, z_hi))
            mouse.scroll_delta = 0
            base._update_goal_marker()

        if mouse.click_pending or (mouse.follow_mode and mouse.inside_top):
            result = _pixel_to_world_xy(
                mouse.last_px, mouse.last_py,
                base.model, base.data,
                cam_name="cf_top",
                render_w=tile_w, render_h=tile_h,
                target_z=float(base._goal_pos[2]),
            )
            if result is not None:
                gx, gy = result
                gx = float(np.clip(gx, -xy_lim, xy_lim))
                gy = float(np.clip(gy, -xy_lim, xy_lim))
                base._goal_pos[0] = gx
                base._goal_pos[1] = gy
                base._update_goal_marker()
                base._reach_hold_counter = 0
                if mouse.click_pending:
                    auto_orbit = False
                    print(f"[eval] Mouse-placed goal: "
                          f"[{gx:+.2f},{gy:+.2f},{base._goal_pos[2]:+.2f}]")
            mouse.click_pending = False

        # --- Auto-orbit ---
        if auto_orbit:
            dt_orbit = max(dt_real, ctrl_dt)
            orbit_phase += orbit_speed * dt_orbit
            r = 0.35
            base._goal_pos[0] = r * math.cos(orbit_phase)
            base._goal_pos[1] = r * math.sin(orbit_phase)
            base._update_goal_marker()

        # ── Physics step (rate-limited to real-time) ──
        if need_step:
            last_tick = now

            if not paused and not ep_done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, rewards, dones, infos = vec_env.step(action)
                reward = float(rewards[0])
                done = bool(dones[0])
                info = infos[0]
                last_info = info
                # Capture raw obs (pre-normalisation) for display
                last_raw_obs = base._observe()
                last_norm_obs = obs[0] if obs.ndim > 1 else obs

                ep_return += reward
                ep_steps += 1

                if args.print_every > 0 and ep_steps % args.print_every == 0:
                    dist = float(info.get("goal_dist", info.get("pos_error", 0)))
                    tilt = float(info.get("tilt_deg", 0))
                    goals = int(info.get("goals_reached", 0))
                    min_rng = float(info.get("min_obstacle_range_m", -1))
                    n_walls = int(info.get("n_active_walls", 0))
                    print(
                        f"[eval] step={ep_steps:5d}  "
                        f"r={reward:+.3f}  dist={dist:.3f}m  "
                        f"tilt={tilt:.1f}°  goals={goals}  "
                        f"walls={n_walls}  min_range={min_rng:.2f}m  "
                        f"goal=[{base._goal_pos[0]:+.2f},"
                        f"{base._goal_pos[1]:+.2f},"
                        f"{base._goal_pos[2]:+.2f}]"
                    )

                if done:
                    n_episodes += 1
                    ep_done = True
                    reason = []
                    for tag in ("crashed", "excessive_tilt", "out_of_bounds",
                                "time_out", "battery_depleted",
                                "obstacle_collision"):
                        if info.get(tag):
                            reason.append(tag)
                    print(
                        f"[eval] Episode {n_episodes} done: "
                        f"return={ep_return:+.2f}  steps={ep_steps}  "
                        f"reason={','.join(reason) or '?'}  "
                        f"goals={int(info.get('goals_reached', 0))}"
                    )
                    print("[eval] Press SPACE to reset.")

        # ── Render & display ──
        frame = base.render(mode="rgb_array")
        hud_info = dict(last_info)
        hud_info["_mouse_follow"] = mouse.follow_mode
        frame = _draw_hud(
            frame,
            goal=base._goal_pos,
            ep_steps=ep_steps,
            ep_return=ep_return,
            goal_speed=goal_speed,
            paused=paused,
            auto_orbit=auto_orbit,
            ep_done=ep_done,
            info=hud_info,
        )
        frame = _draw_nn_inputs(
            frame,
            raw_obs=last_raw_obs,
            norm_obs=last_norm_obs,
            n_lidar=base._n_rf_total,
        )
        # OpenCV uses BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if args.window_scale != 1.0:
            frame_bgr = cv2.resize(
                frame_bgr, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR
            )

        cv2.imshow(_WINDOW_NAME, frame_bgr)

        # Detect window closed via X button
        try:
            if cv2.getWindowProperty(_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                running = False
        except cv2.error:
            running = False

    # ---- Cleanup ----
    cv2.destroyAllWindows()
    vec_env.close()
    print("[eval] Done. Bye!")


if __name__ == "__main__":
    main()
