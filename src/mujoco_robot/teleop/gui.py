"""GUI teleop controller for UR robot arms.

A tkinter-based graphical interface with:
- Directional pad (±X, ±Y) arrow buttons
- Z up/down buttons
- Yaw ±rotation buttons
- Coordinate frame toggle (Base / End-Effector)
- Live MuJoCo camera view
- Joint angle bar displays with numeric readouts
- End-effector position & yaw readout
- Grip toggle (for slot-sorter)
- Reset / Emergency stop buttons
- Speed slider

Usage::

    from mujoco_robot.envs import URReachEnv
    from mujoco_robot.teleop.gui import GUITeleop

    env = URReachEnv(robot="ur5e", time_limit=0)
    GUITeleop(env).run()

Or via CLI::

    python -m mujoco_robot.scripts.teleop --task reach --robot ur5e --gui
"""
from __future__ import annotations

import math
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Optional

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageTk  # type: ignore


# Coordinate frame modes
FRAME_BASE = "base"
FRAME_EE = "ee"


class GUITeleop:
    """Tkinter GUI controller for UR robot environments.

    Parameters
    ----------
    env
        A URReachEnv or URSlotSorterEnv instance.
    task : str
        ``"reach"`` or ``"slot_sorter"`` — controls which buttons are shown.
    fps : int
        Target rendering frame rate.
    """

    # Button colours
    _BTN_BG = "#dce6f0"
    _BTN_ACTIVE = "#b3cde3"
    _BTN_STOP = "#e74c3c"
    _BTN_RESET = "#27ae60"
    _BTN_GRIP = "#8e44ad"
    _PANEL_BG = "#f4f6f8"
    _HEADER_FG = "#2c3e50"

    def __init__(self, env, task: str = "reach", fps: int = 30) -> None:
        self.env = env
        self.task = task
        self.fps = fps
        self.action = np.zeros(env.action_dim, dtype=float)
        self.speed = 0.7
        self._running = False
        self._grip_on = False
        self._reset_pending = False  # consumed by sim thread
        self._frame = FRAME_BASE  # coordinate frame for movement commands

        # Raw directional input (before frame transform)
        self._raw_input = np.zeros(4, dtype=float)  # [x, y, z, yaw]

        # Go-to-point autonomous drive
        self._goto_active = False
        self._goto_pos: Optional[np.ndarray] = None  # target XYZ
        self._goto_yaw: Optional[float] = None        # target yaw (rad)
        self._goto_threshold = 0.02   # position tolerance (m)
        self._goto_yaw_threshold = 0.15  # yaw tolerance (rad)

        # Will be set in _build_ui
        self._root: Optional[tk.Tk] = None
        self._camera_label: Optional[tk.Label] = None
        self._joint_bars: list[ttk.Progressbar] = []
        self._joint_labels: list[tk.Label] = []
        self._ee_labels: dict[str, tk.Label] = {}
        self._status_var: Optional[tk.StringVar] = None
        self._speed_var: Optional[tk.DoubleVar] = None
        self._frame_var: Optional[tk.StringVar] = None
        self._frame_label_var: Optional[tk.StringVar] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._goto_entries: dict[str, tk.Entry] = {}
        self._goto_status_var: Optional[tk.StringVar] = None

        # Click-to-target tracking
        self._click_px: Optional[tuple[int, int]] = None  # last click pixel
        self._click_world: Optional[np.ndarray] = None     # last click 3D pos

        # Available cameras and active selection
        self._cameras = self._detect_cameras()
        self._camera_name = self._cameras[0] if self._cameras else "top"
        self._camera_var: Optional[tk.StringVar] = None

        # Render dimensions (keep in sync with Renderer)
        self._render_w = 540
        self._render_h = 400

        # Create single renderer for GUI view
        self._renderer = mujoco.Renderer(
            self.env.model, height=self._render_h, width=self._render_w
        )

    @staticmethod
    def _detect_cameras() -> list[str]:
        """Return a fixed list of camera names available in the scene."""
        return ["top", "side", "ee_cam"]

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> tk.Tk:
        root = tk.Tk()
        root.title(f"MuJoCo Robot — {self.env.robot.upper()} GUI Teleop")
        root.configure(bg=self._PANEL_BG)
        root.resizable(False, False)

        # ── Main layout: left (camera + info) | right (controls) ──
        left_frame = tk.Frame(root, bg=self._PANEL_BG)
        left_frame.pack(side=tk.LEFT, padx=8, pady=8, fill=tk.BOTH)

        right_frame = tk.Frame(root, bg=self._PANEL_BG)
        right_frame.pack(side=tk.RIGHT, padx=8, pady=8, fill=tk.BOTH)

        # ── Camera selector + label ──
        cam_row = tk.Frame(left_frame, bg=self._PANEL_BG)
        cam_row.pack(fill=tk.X, pady=(0, 4))

        tk.Label(cam_row, text="📷 Camera:", font=("Segoe UI", 11, "bold"),
                 bg=self._PANEL_BG, fg=self._HEADER_FG).pack(side=tk.LEFT)

        self._camera_var = tk.StringVar(value=self._camera_name)
        cam_menu = tk.OptionMenu(cam_row, self._camera_var,
                                 *self._cameras,
                                 command=self._on_camera_change)
        cam_menu.configure(font=("Segoe UI", 10), bg=self._BTN_BG,
                           activebackground=self._BTN_ACTIVE, width=10)
        cam_menu.pack(side=tk.LEFT, padx=(6, 0))

        self._camera_label = tk.Label(left_frame, bg="black",
                                       width=self._render_w,
                                       height=self._render_h,
                                       cursor="crosshair")
        self._camera_label.pack()
        self._camera_label.bind("<Button-1>", self._on_camera_click)
        self._camera_label.bind("<Button-3>", self._on_camera_right_click)
        self._camera_label.bind("<MouseWheel>", self._on_camera_scroll)

        # Hint label
        tk.Label(left_frame,
                 text=("L-click: XY target | Shift+click: XYZ target"
                       " | R-click: Z only | Scroll: adjust Z"),
                 font=("Segoe UI", 8, "italic"),
                 bg=self._PANEL_BG, fg="#95a5a6").pack(anchor="w")

        # ── Info panel (below camera) ──
        info_frame = tk.LabelFrame(left_frame, text="  End-Effector  ",
                                   font=("Segoe UI", 10, "bold"),
                                   bg=self._PANEL_BG, fg=self._HEADER_FG,
                                   padx=8, pady=4)
        info_frame.pack(fill=tk.X, pady=(8, 0))

        self._ee_labels = {}
        for label_text in ["X", "Y", "Z", "Yaw"]:
            row = tk.Frame(info_frame, bg=self._PANEL_BG)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=f"{label_text}:", width=5, anchor="w",
                     font=("Consolas", 10), bg=self._PANEL_BG).pack(side=tk.LEFT)
            val = tk.Label(row, text="0.0000", width=10, anchor="w",
                           font=("Consolas", 10, "bold"), bg=self._PANEL_BG, fg="#2980b9")
            val.pack(side=tk.LEFT)
            self._ee_labels[label_text] = val

        # ── Status bar ──
        self._status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(left_frame, textvariable=self._status_var,
                              font=("Segoe UI", 9), bg="#ecf0f1", fg="#7f8c8d",
                              anchor="w", padx=6, pady=2, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(6, 0))

        # ── Right panel: controls ──
        self._build_controls(right_frame)

        return root

    def _build_controls(self, parent: tk.Frame) -> None:
        """Build the control buttons and joint displays."""

        # ── Speed slider ──
        speed_frame = tk.LabelFrame(parent, text="  Speed  ",
                                    font=("Segoe UI", 10, "bold"),
                                    bg=self._PANEL_BG, fg=self._HEADER_FG,
                                    padx=8, pady=4)
        speed_frame.pack(fill=tk.X, pady=(0, 8))

        self._speed_var = tk.DoubleVar(value=self.speed)
        speed_scale = tk.Scale(speed_frame, from_=0.1, to=1.0, resolution=0.05,
                               orient=tk.HORIZONTAL, variable=self._speed_var,
                               command=self._on_speed_change,
                               bg=self._PANEL_BG, highlightthickness=0, length=200)
        speed_scale.pack(fill=tk.X)

        # ── Coordinate frame toggle ──
        frame_outer = tk.LabelFrame(parent, text="  Coordinate Frame  ",
                                    font=("Segoe UI", 10, "bold"),
                                    bg=self._PANEL_BG, fg=self._HEADER_FG,
                                    padx=8, pady=6)
        frame_outer.pack(fill=tk.X, pady=(0, 8))

        self._frame_var = tk.StringVar(value=FRAME_BASE)
        self._frame_label_var = tk.StringVar(value="🌐 BASE (World)")

        frame_row = tk.Frame(frame_outer, bg=self._PANEL_BG)
        frame_row.pack(fill=tk.X)

        rb_base = tk.Radiobutton(frame_row, text="🌐 Base (World)",
                                 variable=self._frame_var, value=FRAME_BASE,
                                 command=self._on_frame_change,
                                 font=("Segoe UI", 10), bg=self._PANEL_BG,
                                 activebackground=self._PANEL_BG,
                                 selectcolor=self._PANEL_BG)
        rb_base.pack(side=tk.LEFT, padx=(0, 8))

        rb_ee = tk.Radiobutton(frame_row, text="🔧 End-Effector (Tool)",
                               variable=self._frame_var, value=FRAME_EE,
                               command=self._on_frame_change,
                               font=("Segoe UI", 10), bg=self._PANEL_BG,
                               activebackground=self._PANEL_BG,
                               selectcolor=self._PANEL_BG)
        rb_ee.pack(side=tk.LEFT)

        # Frame indicator label — shows which axes are active
        self._frame_info = tk.Label(frame_outer, textvariable=self._frame_label_var,
                                    font=("Segoe UI", 9, "italic"),
                                    bg=self._PANEL_BG, fg="#7f8c8d")
        self._frame_info.pack(pady=(2, 0))

        # ── D-pad: XY movement ──
        dpad_frame = tk.LabelFrame(parent, text="  XY Movement  ",
                                   font=("Segoe UI", 10, "bold"),
                                   bg=self._PANEL_BG, fg=self._HEADER_FG,
                                   padx=12, pady=8)
        dpad_frame.pack(fill=tk.X, pady=(0, 6))

        dpad_grid = tk.Frame(dpad_frame, bg=self._PANEL_BG)
        dpad_grid.pack()

        btn_kw = dict(width=4, height=2, font=("Segoe UI", 12, "bold"),
                      bg=self._BTN_BG, activebackground=self._BTN_ACTIVE, relief=tk.RAISED, bd=2)

        # Row 0: forward (+Y in base, +local-X in EE)
        self._btn_fwd = tk.Button(dpad_grid, text="▲\n+Y", **btn_kw)
        self._btn_fwd.grid(row=0, column=1, padx=2, pady=2)
        self._bind_hold(self._btn_fwd, lambda: self._set_raw(1, 1.0), lambda: self._set_raw(1, 0.0))

        # Row 1: left (-X), stop, right (+X)
        self._btn_left = tk.Button(dpad_grid, text="◄\n-X", **btn_kw)
        self._btn_left.grid(row=1, column=0, padx=2, pady=2)
        self._bind_hold(self._btn_left, lambda: self._set_raw(0, -1.0), lambda: self._set_raw(0, 0.0))

        btn_stop_xy = tk.Button(dpad_grid, text="■\nSTOP", width=4, height=2,
                                font=("Segoe UI", 10, "bold"),
                                bg="#f39c12", activebackground="#e67e22", relief=tk.RAISED, bd=2,
                                command=self._stop_all)
        btn_stop_xy.grid(row=1, column=1, padx=2, pady=2)

        self._btn_right = tk.Button(dpad_grid, text="►\n+X", **btn_kw)
        self._btn_right.grid(row=1, column=2, padx=2, pady=2)
        self._bind_hold(self._btn_right, lambda: self._set_raw(0, 1.0), lambda: self._set_raw(0, 0.0))

        # Row 2: backward (-Y)
        self._btn_back = tk.Button(dpad_grid, text="▼\n-Y", **btn_kw)
        self._btn_back.grid(row=2, column=1, padx=2, pady=2)
        self._bind_hold(self._btn_back, lambda: self._set_raw(1, -1.0), lambda: self._set_raw(1, 0.0))

        # ── Z movement ──
        z_frame = tk.LabelFrame(parent, text="  Z Movement  ",
                                font=("Segoe UI", 10, "bold"),
                                bg=self._PANEL_BG, fg=self._HEADER_FG,
                                padx=12, pady=8)
        z_frame.pack(fill=tk.X, pady=(0, 6))

        z_grid = tk.Frame(z_frame, bg=self._PANEL_BG)
        z_grid.pack()

        btn_up = tk.Button(z_grid, text="⬆\nZ+", **btn_kw)
        btn_up.grid(row=0, column=0, padx=4, pady=2)
        self._bind_hold(btn_up, lambda: self._set_raw(2, 1.0), lambda: self._set_raw(2, 0.0))

        btn_down = tk.Button(z_grid, text="⬇\nZ-", **btn_kw)
        btn_down.grid(row=0, column=1, padx=4, pady=2)
        self._bind_hold(btn_down, lambda: self._set_raw(2, -1.0), lambda: self._set_raw(2, 0.0))

        # ── Yaw rotation ──
        yaw_frame = tk.LabelFrame(parent, text="  Yaw Rotation  ",
                                  font=("Segoe UI", 10, "bold"),
                                  bg=self._PANEL_BG, fg=self._HEADER_FG,
                                  padx=12, pady=8)
        yaw_frame.pack(fill=tk.X, pady=(0, 6))

        yaw_grid = tk.Frame(yaw_frame, bg=self._PANEL_BG)
        yaw_grid.pack()

        btn_yaw_l = tk.Button(yaw_grid, text="↺\nCCW", **btn_kw)
        btn_yaw_l.grid(row=0, column=0, padx=4, pady=2)
        self._bind_hold(btn_yaw_l, lambda: self._set_raw(3, 1.0), lambda: self._set_raw(3, 0.0))

        btn_yaw_r = tk.Button(yaw_grid, text="↻\nCW", **btn_kw)
        btn_yaw_r.grid(row=0, column=1, padx=4, pady=2)
        self._bind_hold(btn_yaw_r, lambda: self._set_raw(3, -1.0), lambda: self._set_raw(3, 0.0))

        # ── Grip button (slot_sorter only) ──
        if self.task == "slot_sorter" and self.env.action_dim >= 5:
            grip_frame = tk.Frame(parent, bg=self._PANEL_BG)
            grip_frame.pack(fill=tk.X, pady=(0, 6))
            self._grip_btn = tk.Button(grip_frame, text="🤏 GRIP: OFF",
                                       width=20, height=2,
                                       font=("Segoe UI", 11, "bold"),
                                       bg=self._BTN_GRIP, fg="white",
                                       activebackground="#9b59b6",
                                       command=self._toggle_grip)
            self._grip_btn.pack()

        # ── Action buttons ──
        action_frame = tk.Frame(parent, bg=self._PANEL_BG)
        action_frame.pack(fill=tk.X, pady=(4, 8))

        tk.Button(action_frame, text="🔄 RESET", width=10, height=2,
                  font=("Segoe UI", 10, "bold"), bg=self._BTN_RESET, fg="white",
                  activebackground="#2ecc71", command=self._reset).pack(side=tk.LEFT, padx=4)

        tk.Button(action_frame, text="🛑 E-STOP", width=10, height=2,
                  font=("Segoe UI", 10, "bold"), bg=self._BTN_STOP, fg="white",
                  activebackground="#c0392b", command=self._stop_all).pack(side=tk.LEFT, padx=4)

        # ── Go To Point ──
        goto_frame = tk.LabelFrame(parent, text="  Go To Point  ",
                                   font=("Segoe UI", 10, "bold"),
                                   bg=self._PANEL_BG, fg=self._HEADER_FG,
                                   padx=8, pady=6)
        goto_frame.pack(fill=tk.X, pady=(0, 6))

        # Pre-fill with current EE position
        ee_pos = self.env.data.site_xpos[self.env.ee_site]
        ee_yaw_deg = math.degrees(self.env._ee_yaw())
        defaults = {"X": f"{ee_pos[0]:.3f}", "Y": f"{ee_pos[1]:.3f}",
                    "Z": f"{ee_pos[2]:.3f}", "Yaw": f"{ee_yaw_deg:.1f}"}

        self._goto_entries = {}
        for label_text in ["X", "Y", "Z", "Yaw"]:
            row = tk.Frame(goto_frame, bg=self._PANEL_BG)
            row.pack(fill=tk.X, pady=1)
            unit = "°" if label_text == "Yaw" else "m"
            tk.Label(row, text=f"{label_text} ({unit}):", width=7, anchor="w",
                     font=("Consolas", 9), bg=self._PANEL_BG).pack(side=tk.LEFT)
            entry = tk.Entry(row, width=10, font=("Consolas", 9))
            entry.insert(0, defaults[label_text])
            entry.pack(side=tk.LEFT, padx=(2, 0))
            self._goto_entries[label_text] = entry

        goto_btn_row = tk.Frame(goto_frame, bg=self._PANEL_BG)
        goto_btn_row.pack(fill=tk.X, pady=(4, 0))

        tk.Button(goto_btn_row, text="\u25B6 GO", width=7, height=1,
                  font=("Segoe UI", 10, "bold"),
                  bg="#2980b9", fg="white", activebackground="#3498db",
                  command=self._start_goto).pack(side=tk.LEFT, padx=2)

        tk.Button(goto_btn_row, text="\U0001F3B2 Random", width=9, height=1,
                  font=("Segoe UI", 10, "bold"),
                  bg="#e67e22", fg="white", activebackground="#f39c12",
                  command=self._random_goal).pack(side=tk.LEFT, padx=2)

        tk.Button(goto_btn_row, text="\u23F9 Stop", width=7, height=1,
                  font=("Segoe UI", 10, "bold"),
                  bg="#95a5a6", fg="white", activebackground="#bdc3c7",
                  command=self._stop_goto).pack(side=tk.LEFT, padx=2)

        self._goto_status_var = tk.StringVar(value="Idle")
        tk.Label(goto_frame, textvariable=self._goto_status_var,
                 font=("Segoe UI", 9, "italic"),
                 bg=self._PANEL_BG, fg="#7f8c8d").pack(pady=(2, 0))

        # ── Joint angles display ──
        joint_frame = tk.LabelFrame(parent, text="  Joint Angles (deg)  ",
                                    font=("Segoe UI", 10, "bold"),
                                    bg=self._PANEL_BG, fg=self._HEADER_FG,
                                    padx=8, pady=4)
        joint_frame.pack(fill=tk.X, pady=(0, 4))

        joint_names = ["Shoulder Pan", "Shoulder Lift", "Elbow",
                       "Wrist 1", "Wrist 2", "Wrist 3"]
        self._joint_bars = []
        self._joint_labels = []

        for i, jname in enumerate(joint_names):
            row = tk.Frame(joint_frame, bg=self._PANEL_BG)
            row.pack(fill=tk.X, pady=1)

            tk.Label(row, text=f"J{i+1}", width=3, anchor="w",
                     font=("Consolas", 9, "bold"), bg=self._PANEL_BG,
                     fg="#34495e").pack(side=tk.LEFT)

            bar = ttk.Progressbar(row, orient=tk.HORIZONTAL, length=120,
                                  mode="determinate", maximum=360)
            bar.pack(side=tk.LEFT, padx=(2, 4))
            self._joint_bars.append(bar)

            val_label = tk.Label(row, text="  0.0°", width=8, anchor="e",
                                 font=("Consolas", 9), bg=self._PANEL_BG, fg="#2c3e50")
            val_label.pack(side=tk.LEFT)
            self._joint_labels.append(val_label)

    # ---------------------------------------------------------------- Bind helpers
    def _bind_hold(self, btn: tk.Button, on_press, on_release) -> None:
        """Bind press-and-hold: fire action on press, stop on release."""
        btn.bind("<ButtonPress-1>", lambda e: on_press())
        btn.bind("<ButtonRelease-1>", lambda e: on_release())

    def _set_raw(self, axis: int, value: float) -> None:
        """Set raw directional input. Frame transform is applied in _apply_frame."""
        self._raw_input[axis] = value

    def _apply_frame(self) -> None:
        """Transform raw input [x, y, z, yaw] through the active coordinate frame.

        - **Base frame**: raw input maps directly to world XYZ + yaw.
        - **EE frame**: "Forward" (raw axis 1) drives along the tool's local
          X-axis, "Left" (raw axis 0, negative) drives along the tool's
          local Y-axis.  The tool's local axes in the world are:
              local X  →  [cos(yaw), sin(yaw)]
              local Y  →  [-sin(yaw), cos(yaw)]
          Z and yaw pass through unchanged.
        """
        raw = self._raw_input * self.speed

        if self._frame == FRAME_EE:
            yaw = self.env._ee_yaw()
            c, s = math.cos(yaw), math.sin(yaw)
            # Map GUI inputs to tool-frame directions:
            #   raw[1] = fwd(+1)/back(-1)  →  tool local-X  = [ cos, sin]
            #   raw[0] = right(+1)/left(-1) → but "left" should move along
            #            tool local+Y = [-sin, cos], so we negate raw[0].
            fwd  = raw[1]    # along tool X
            left = -raw[0]   # along tool Y  (left button → +1 here)
            self.action[0] = fwd * c - left * s
            self.action[1] = fwd * s + left * c
        else:
            # Base frame — pass through
            self.action[0] = raw[0]
            self.action[1] = raw[1]

        # Z and yaw are the same in both frames
        self.action[2] = raw[2]
        self.action[3] = raw[3]

    def _stop_all(self) -> None:
        self._raw_input[:] = 0.0
        self.action[:] = 0.0
        self._grip_on = False
        self._goto_active = False

    def _reset(self) -> None:
        """Request a reset — the sim thread will execute it safely."""
        self._raw_input[:] = 0.0
        self.action[:] = 0.0
        self._grip_on = False
        self._goto_active = False
        self._reset_pending = True

    def _toggle_grip(self) -> None:
        self._grip_on = not self._grip_on
        if self.env.action_dim >= 5:
            self.action[4] = 1.0 if self._grip_on else 0.0
        if hasattr(self, "_grip_btn"):
            state = "ON" if self._grip_on else "OFF"
            self._grip_btn.configure(text=f"🤏 GRIP: {state}")

    def _on_speed_change(self, val) -> None:
        self.speed = float(val)

    def _on_camera_change(self, cam_name: str) -> None:
        """Handle camera dropdown selection."""
        self._camera_name = cam_name
        if self._camera_var:
            self._camera_var.set(cam_name)

    def _cycle_camera(self) -> None:
        """Cycle to the next camera in the list."""
        idx = self._cameras.index(self._camera_name)
        self._camera_name = self._cameras[(idx + 1) % len(self._cameras)]
        if self._camera_var:
            self._camera_var.set(self._camera_name)

    def _on_frame_change(self) -> None:
        """Handle coordinate frame radio button change."""
        self._frame = self._frame_var.get()
        if self._frame == FRAME_BASE:
            self._frame_label_var.set("XY = World axes  |  Z = Up  |  Yaw = World Z")
            self._btn_fwd.configure(text="▲\n+Y")
            self._btn_back.configure(text="▼\n-Y")
            self._btn_left.configure(text="◄\n-X")
            self._btn_right.configure(text="►\n+X")
        else:
            self._frame_label_var.set("XY = Tool axes  |  Z = Up  |  Yaw = World Z")
            self._btn_fwd.configure(text="▲\nFwd")
            self._btn_back.configure(text="▼\nBack")
            self._btn_left.configure(text="◄\nLeft")
            self._btn_right.configure(text="►\nRight")

    # ----------------------------------------------------------- Go-To-Point
    def _start_goto(self) -> None:
        """Parse the entry fields and begin autonomous IK drive."""
        try:
            x = float(self._goto_entries["X"].get())
            y = float(self._goto_entries["Y"].get())
            z = float(self._goto_entries["Z"].get())
            yaw_deg = float(self._goto_entries["Yaw"].get())
        except ValueError:
            if self._goto_status_var:
                self._goto_status_var.set("Invalid input — use numbers")
            return

        self._goto_pos = np.array([x, y, z])
        self._goto_yaw = math.radians(yaw_deg)

        # Place the goal marker at the target
        self.env.goal_pos = self._goto_pos.copy()
        self.env.goal_yaw = self._goto_yaw
        self.env._place_goal_marker(self.env.goal_pos, self.env.goal_yaw)

        # Stop any manual input and activate autonomous drive
        self._raw_input[:] = 0.0
        self.action[:] = 0.0
        self._goto_active = True

        if self._goto_status_var:
            self._goto_status_var.set(
                f"Driving to ({x:.3f}, {y:.3f}, {z:.3f}) {yaw_deg:.0f}°"
            )

    def _stop_goto(self) -> None:
        """Cancel autonomous drive."""
        self._goto_active = False
        self.action[:] = 0.0
        if self._goto_status_var:
            self._goto_status_var.set("Stopped")

    def _random_goal(self) -> None:
        """Sample a new random goal, fill the entry fields, and start driving."""
        goal = self.env._sample_goal()
        goal_yaw = float(self.env._rng.uniform(-math.pi, math.pi))

        self._fill_goto_entries(goal, goal_yaw)

        # Place marker
        self.env.goal_pos = goal.copy()
        self.env.goal_yaw = goal_yaw
        self.env._place_goal_marker(goal, goal_yaw)

        # Start driving
        self._goto_pos = goal.copy()
        self._goto_yaw = goal_yaw
        self._raw_input[:] = 0.0
        self.action[:] = 0.0
        self._goto_active = True

        if self._goto_status_var:
            self._goto_status_var.set(
                f"Random goal — driving to "
                f"({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f})"
            )

    def _fill_goto_entries(self, pos: np.ndarray, yaw: float) -> None:
        """Fill the XYZ + Yaw entry fields with the given values."""
        vals = {"X": f"{pos[0]:.3f}", "Y": f"{pos[1]:.3f}",
                "Z": f"{pos[2]:.3f}", "Yaw": f"{math.degrees(yaw):.1f}"}
        for key, entry in self._goto_entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, vals[key])

    # --------------------------------------------------------- Click-to-target
    def _pixel_to_world(
        self, px: int, py: int
    ) -> tuple[Optional[np.ndarray], Optional[str]]:
        """Ray-cast a pixel in the camera view into 3D world coordinates.

        Returns ``(position, geom_name)`` or ``(None, None)`` if no hit.
        When the ray hits the table or floor, only XY is used and Z is
        kept at the current EE height (more useful than slamming down).
        """
        model, data = self.env.model, self.env.data
        cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, self._camera_name
        )
        if cam_id < 0:
            return None, None

        cam_pos = data.cam_xpos[cam_id].copy()
        cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
        fovy = math.radians(model.cam_fovy[cam_id])
        aspect = self._render_w / self._render_h

        half_h = math.tan(fovy / 2)
        half_w = half_h * aspect

        # NDC: image (0,0) = top-left
        ndc_x = 2.0 * px / self._render_w - 1.0
        ndc_y = 1.0 - 2.0 * py / self._render_h

        dir_cam = np.array([ndc_x * half_w, ndc_y * half_h, -1.0])
        dir_cam /= np.linalg.norm(dir_cam)
        dir_world = cam_mat @ dir_cam

        geomid = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(
            model, data,
            pnt=cam_pos, vec=dir_world,
            geomgroup=None, flg_static=1, bodyexclude=-1,
            geomid=geomid,
        )

        if dist >= 0:
            hit = cam_pos + dist * dir_world
            gname = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, int(geomid[0])
            )
            return hit, gname

        # Fallback: project onto table plane (z = table top)
        table_z = 0.74
        if abs(dir_world[2]) > 1e-6:
            t = (table_z - cam_pos[2]) / dir_world[2]
            if t > 0:
                return cam_pos + t * dir_world, "table_plane"
        return None, None

    def _on_camera_click(self, event: tk.Event) -> None:
        """Left-click on the camera view.

        * **Plain click** — set XY from the hit point; Z stays at the
          current EE height (useful for table / floor hits).
        * **Shift+click** — use the full 3-D hit point including Z
          (useful for clicking on elevated surfaces or objects).
        """
        px, py = event.x, event.y
        hit_pos, geom_name = self._pixel_to_world(px, py)
        if hit_pos is None:
            return

        shift_held = bool(event.state & 0x0001)  # Shift modifier

        if not shift_held:
            # Plain click → keep current EE Z
            ee_z = self.env.data.site_xpos[self.env.ee_site][2]
            hit_pos[2] = ee_z

        self._activate_goto_from_click(px, py, hit_pos)

    def _on_camera_right_click(self, event: tk.Event) -> None:
        """Right-click on the camera view — adjust Z (height) only.

        The ray hit's Z coordinate becomes the new target Z while XY
        are kept from the current goto target (or the current EE pos
        if no target is active).
        """
        px, py = event.x, event.y
        hit_pos, _geom_name = self._pixel_to_world(px, py)
        if hit_pos is None:
            return

        # Start from the existing goto XY, or the current EE position
        if self._goto_pos is not None:
            target = self._goto_pos.copy()
        else:
            target = self.env.data.site_xpos[self.env.ee_site].copy()

        target[2] = hit_pos[2]  # only update Z

        self._activate_goto_from_click(px, py, target)

    def _on_camera_scroll(self, event: tk.Event) -> None:
        """Scroll wheel on the camera view — fine-adjust target Z.

        Each scroll tick moves the target height by ±1 cm.
        Shift+scroll uses ±5 cm for coarser adjustment.
        """
        step = 0.05 if (event.state & 0x0001) else 0.01  # Shift → 5 cm
        delta = step if event.delta > 0 else -step

        # Start from existing target or current EE pos
        if self._goto_pos is not None:
            target = self._goto_pos.copy()
        else:
            target = self.env.data.site_xpos[self.env.ee_site].copy()

        target[2] = float(np.clip(target[2] + delta, 0.0, 2.0))

        # Update the goto entries and drive
        current_yaw = (self._goto_yaw
                       if self._goto_yaw is not None
                       else self.env._ee_yaw())
        self._fill_goto_entries(target, current_yaw)

        self.env.goal_pos = target.copy()
        self.env.goal_yaw = current_yaw
        self.env._place_goal_marker(target, current_yaw)

        self._goto_pos = target.copy()
        self._goto_yaw = current_yaw
        self._raw_input[:] = 0.0
        self.action[:] = 0.0
        self._goto_active = True

        if self._goto_status_var:
            self._goto_status_var.set(
                f"Z → {target[2]:.3f} m  "
                f"({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})"
            )

    def _activate_goto_from_click(
        self, px: int, py: int, target_pos: np.ndarray
    ) -> None:
        """Shared helper: set the goto target from a camera-view click."""
        self._click_px = (px, py)
        self._click_world = target_pos.copy()

        current_yaw = self.env._ee_yaw()
        self._fill_goto_entries(target_pos, current_yaw)

        self.env.goal_pos = target_pos.copy()
        self.env.goal_yaw = current_yaw
        self.env._place_goal_marker(target_pos, current_yaw)

        self._goto_pos = target_pos.copy()
        self._goto_yaw = current_yaw
        self._raw_input[:] = 0.0
        self.action[:] = 0.0
        self._goto_active = True

        if self._goto_status_var:
            self._goto_status_var.set(
                f"Click target ({target_pos[0]:.2f}, "
                f"{target_pos[1]:.2f}, {target_pos[2]:.2f})"
            )

    def _compute_goto_action(self) -> Optional[np.ndarray]:
        """Compute a [-1, 1] action vector that drives the EE toward the target.

        Uses a proportional controller with damping near the goal to
        prevent oscillation.  Returns ``None`` when the target is reached.
        """
        ee_pos = self.env.data.site_xpos[self.env.ee_site].copy()
        ee_yaw = self.env._ee_yaw()

        pos_err = self._goto_pos - ee_pos
        pos_dist = float(np.linalg.norm(pos_err))

        # Yaw error wrapped to [-pi, pi]
        yaw_err = self._goto_yaw - ee_yaw
        yaw_err = float((yaw_err + math.pi) % (2 * math.pi) - math.pi)

        pos_ok = pos_dist < self._goto_threshold
        yaw_ok = abs(yaw_err) < self._goto_yaw_threshold

        if pos_ok and yaw_ok:
            return None  # signal: target reached

        action = np.zeros(self.env.action_dim, dtype=float)

        if not pos_ok:
            # Proportional + damped: ramp linearly from 0.15 to 1.0
            # over the range [threshold .. ee_step*5].  The minimum of
            # 0.15 keeps the action above hold_eps so the IK actually
            # moves, while the upper clamp prevents overshoot.
            ramp = pos_dist / (self.env.ee_step * 5)
            gain = float(np.clip(ramp, 0.15, 1.0))
            direction = pos_err / max(pos_dist, 1e-8)
            action[:3] = direction * gain

        if not yaw_ok:
            yaw_ramp = abs(yaw_err) / (self.env.yaw_step * 3)
            yaw_gain = float(np.clip(yaw_ramp, 0.15, 1.0))
            action[3] = math.copysign(yaw_gain, yaw_err)

        return action

    # ---------------------------------------------------------------- Sim loop
    def _sim_loop(self) -> None:
        """Background simulation loop — steps env and queues UI updates."""
        ctrl_dt = self.env.model.opt.timestep * self.env.n_substeps
        while self._running:
            t0 = time.time()

            # Handle pending reset (from UI thread) safely here
            if self._reset_pending:
                self._reset_pending = False
                self.env.reset()
                self.action[:] = 0.0
                self._raw_input[:] = 0.0
                self._grip_on = False
                self._root.after(0, lambda: self._status_var.set(
                    "Environment reset"
                ))
                continue

            # ── Autonomous go-to-point drive ──
            if self._goto_active:
                # Any manual input cancels autonomous drive
                if np.any(self._raw_input != 0):
                    self._goto_active = False
                    self._root.after(0, lambda: (
                        self._goto_status_var.set("Cancelled (manual input)")
                        if self._goto_status_var else None
                    ))
                else:
                    goto_act = self._compute_goto_action()
                    if goto_act is None:
                        # Target reached
                        self._goto_active = False
                        self.env._goals_reached += 1
                        self._root.after(0, lambda: (
                            self._goto_status_var.set("Target reached!")
                            if self._goto_status_var else None
                        ))
                        self.action[:] = 0.0
                    else:
                        self.env.step(goto_act)

                    elapsed = time.time() - t0
                    time.sleep(max(0, ctrl_dt - elapsed))
                    continue

            # Apply coordinate frame transform before stepping
            self._apply_frame()

            # Pure-yaw shortcut: rotate wrist3 directly to avoid whole-arm
            # movement when the user only presses yaw buttons.
            has_xyz = np.any(self._raw_input[:3] != 0)
            has_yaw = self._raw_input[3] != 0

            if has_yaw and not has_xyz:
                # Direct wrist3 rotation — much cleaner than full IK yaw
                dyaw = self.action[3] * self.env.yaw_step
                self.env._last_targets[5] += dyaw   # wrist3 index
                targets = self.env._clamp_to_limits(self.env._last_targets)
                self.env._last_targets = targets.copy()
                for k, act_id in enumerate(self.env.robot_actuators):
                    self.env.data.ctrl[act_id] = targets[k]
                for _ in range(self.env.n_substeps):
                    mujoco.mj_step(self.env.model, self.env.data)
                # Sync _target_yaw to actual EE yaw so that subsequent
                # IK steps (XY movement) don't fight a stale target.
                self.env._target_yaw = self.env._ee_yaw()
            else:
                # Normal path: full IK step
                self.env.step(self.action)

            elapsed = time.time() - t0
            sleep_time = max(0, ctrl_dt - elapsed)
            time.sleep(sleep_time)

    def _update_display(self) -> None:
        """Periodic UI update — render camera, update joint/EE readouts."""
        if not self._running:
            return

        try:
            # Render camera frame
            self._renderer.update_scene(self.env.data, camera=self._camera_name)
            frame = self._renderer.render()

            # Convert to PIL and draw crosshair overlay if active
            img = Image.fromarray(frame)
            if self._click_px is not None and self._goto_active:
                draw = ImageDraw.Draw(img)
                cx, cy = self._click_px
                r = 10  # crosshair radius
                color = (0, 255, 100)  # bright green
                draw.line([(cx - r, cy), (cx + r, cy)], fill=color, width=2)
                draw.line([(cx, cy - r), (cx, cy + r)], fill=color, width=2)
                draw.ellipse(
                    [(cx - r, cy - r), (cx + r, cy + r)],
                    outline=color, width=2,
                )
            self._photo = ImageTk.PhotoImage(image=img)
            self._camera_label.configure(image=self._photo)

            # Update joint angles
            for i, jname in enumerate(self.env.robot_joints):
                jid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                angle_rad = self.env.data.qpos[self.env.model.jnt_qposadr[jid]]
                angle_deg = math.degrees(angle_rad)
                # Bar: map [-180, 180] → [0, 360]
                self._joint_bars[i]["value"] = angle_deg + 180
                self._joint_labels[i].configure(text=f"{angle_deg:+7.1f}°")

            # Update EE readout
            ee_pos = self.env.data.site_xpos[self.env.ee_site]
            ee_yaw = self.env._ee_yaw()
            self._ee_labels["X"].configure(text=f"{ee_pos[0]:+.4f} m")
            self._ee_labels["Y"].configure(text=f"{ee_pos[1]:+.4f} m")
            self._ee_labels["Z"].configure(text=f"{ee_pos[2]:+.4f} m")
            self._ee_labels["Yaw"].configure(text=f"{math.degrees(ee_yaw):+.1f}°")

            # Status
            dist = float(np.linalg.norm(ee_pos - self.env.goal_pos))
            frame_tag = "BASE" if self._frame == FRAME_BASE else "EE"
            goto_tag = " | AUTO" if self._goto_active else ""
            self._status_var.set(
                f"Frame: {frame_tag}{goto_tag}  |  "
                f"Dist: {dist:.3f} m  |  "
                f"Speed: {self.speed:.1f}  |  "
                f"Goals: {self.env._goals_reached}"
            )

            # Update goto status with live distance
            if self._goto_active and self._goto_pos is not None:
                goto_dist = float(np.linalg.norm(ee_pos - self._goto_pos))
                yaw_err_deg = 0.0
                if self._goto_yaw is not None:
                    raw_err = self._goto_yaw - ee_yaw
                    yaw_err_deg = math.degrees(
                        (raw_err + math.pi) % (2 * math.pi) - math.pi
                    )
                self._goto_status_var.set(
                    f"Driving... dist={goto_dist:.3f} m  "
                    f"yaw_err={yaw_err_deg:+.1f}deg"
                )

        except Exception:
            pass  # renderer may not be ready yet

        # Schedule next update
        interval_ms = max(16, 1000 // self.fps)
        self._root.after(interval_ms, self._update_display)

    # ---------------------------------------------------------------- Keyboard
    def _on_key_press(self, event) -> None:
        key = event.keysym.lower()
        mapping = {
            "w": (1,  1.0), "s": (1, -1.0),
            "a": (0, -1.0), "d": (0,  1.0),
            "r": (2,  1.0), "f": (2, -1.0),
            "q": (3,  1.0), "e": (3, -1.0),
        }
        if key in mapping:
            axis, val = mapping[key]
            self._set_raw(axis, val)
        elif key == "x":
            self._stop_all()
        elif key == "space" and self.task == "slot_sorter":
            self._toggle_grip()
        elif key == "tab":
            # Toggle coordinate frame with Tab key
            if self._frame == FRAME_BASE:
                self._frame_var.set(FRAME_EE)
            else:
                self._frame_var.set(FRAME_BASE)
            self._on_frame_change()
        elif key == "c":
            self._cycle_camera()
        elif key == "g":
            self._start_goto()
        elif key == "n":
            self._random_goal()

    def _on_key_release(self, event) -> None:
        key = event.keysym.lower()
        axis_map = {"w": 1, "s": 1, "a": 0, "d": 0,
                    "r": 2, "f": 2, "q": 3, "e": 3}
        if key in axis_map:
            self._set_raw(axis_map[key], 0.0)

    # ---------------------------------------------------------------- Run
    def run(self) -> None:
        """Launch the GUI — blocks until the window is closed."""
        self._root = self._build_ui()
        self._running = True

        # Keyboard bindings (also work while GUI is focused)
        self._root.bind("<KeyPress>", self._on_key_press)
        self._root.bind("<KeyRelease>", self._on_key_release)

        # Reset env
        self.env.reset()

        # Start simulation in background thread
        sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        sim_thread.start()

        # Start display updates
        self._root.after(100, self._update_display)

        # Run tkinter main loop
        def on_close():
            self._running = False
            time.sleep(0.1)
            try:
                self._renderer.close()
            except Exception:
                pass
            self.env.close()
            self._root.destroy()

        self._root.protocol("WM_DELETE_WINDOW", on_close)
        self._root.mainloop()
