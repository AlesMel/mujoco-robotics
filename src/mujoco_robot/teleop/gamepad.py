"""DualShock gamepad teleop for the slot-sorter environment.

Requires ``pygame`` for gamepad input.

Usage::

    from mujoco_robot.envs import URSlotSorterEnv
    from mujoco_robot.teleop.gamepad import DualShockTeleop

    env = URSlotSorterEnv(time_limit=0)
    DualShockTeleop(env).run()
"""
from __future__ import annotations

import time

import mujoco
import mujoco.viewer
import numpy as np

try:
    import pygame  # type: ignore
except ImportError:
    pygame = None  # graceful degradation


class DualShockTeleop:
    """DualShock / DualSense gamepad teleop for the slot-sorter.

    Controls:
        Left stick  — move XY
        Right stick Y — move Z
        L2 / R2     — yaw
        X (cross)   — toggle grip
        Triangle    — reset

    Parameters
    ----------
    env : URSlotSorterEnv
        Slot-sorter environment instance.
    """

    DEADZONE = 0.12

    def __init__(self, env) -> None:
        if pygame is None:
            raise ImportError("pygame is required for gamepad teleop. "
                              "Install it with: pip install pygame")
        self.env = env
        self.action = np.zeros(env.action_dim, dtype=float)
        self.speed_xy = 1.5
        self.speed_z = 1.5
        self.speed_yaw = 1.2

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected")
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()
        print(f"Gamepad: {self.joy.get_name()} "
              f"({self.joy.get_numaxes()} axes, {self.joy.get_numbuttons()} buttons)")

    @staticmethod
    def _apply_deadzone(val: float, dz: float) -> float:
        if abs(val) < dz:
            return 0.0
        sign = 1.0 if val > 0 else -1.0
        return sign * (abs(val) - dz) / (1.0 - dz)

    def run(self) -> None:
        """Launch the interactive viewer + gamepad loop."""
        obs = self.env.reset()
        ctrl_period = self.env.model.opt.timestep * self.env.n_substeps
        print(f"Gamepad control ({ctrl_period*1000:.0f} ms/step):")
        print("  LS  -> move XY")
        print("  RS Y -> move Z")
        print("  L2/R2 -> yaw")
        print("  X (button 0) -> toggle grip")
        print("  Triangle (button 3) -> reset")

        grip_closed = False
        info = {"successes": 0, "grasped": -1, "n_objects": self.env.n_objects}

        with mujoco.viewer.launch_passive(self.env.model, self.env.data) as viewer:
            last = time.time()
            while viewer.is_running():
                for event in pygame.event.get():
                    if event.type == pygame.JOYBUTTONDOWN:
                        if event.button == 0:  # X / Cross
                            grip_closed = not grip_closed
                            self.action[4] = 1.0 if grip_closed else 0.0
                        elif event.button == 3:  # Triangle
                            self.env.reset()
                            self.action[:] = 0
                            grip_closed = False
                            info = {"successes": 0, "grasped": -1,
                                    "n_objects": self.env.n_objects}

                lx = self._apply_deadzone(self.joy.get_axis(0), self.DEADZONE)
                ly = self._apply_deadzone(self.joy.get_axis(1), self.DEADZONE)
                ry = (self._apply_deadzone(self.joy.get_axis(4), self.DEADZONE)
                      if self.joy.get_numaxes() > 4 else 0.0)
                r2 = (max(0.0, (self.joy.get_axis(5) + 1) / 2)
                      if self.joy.get_numaxes() > 5 else 0.0)
                l2 = (max(0.0, (self.joy.get_axis(2) + 1) / 2)
                      if self.joy.get_numaxes() > 2 else 0.0)

                self.action[0] = lx * self.speed_xy
                self.action[1] = -ly * self.speed_xy
                self.action[2] = -ry * self.speed_z
                self.action[3] = (r2 - l2) * self.speed_yaw

                now = time.time()
                if now - last < ctrl_period:
                    time.sleep(0.001)
                    continue
                last = now

                step = self.env.step(self.action)
                info = step.info
                viewer.sync()

                status = "".join(
                    "#" if s else "_"
                    for s in self.env.success_mask[: self.env.n_objects]
                )
                grasp_str = "none" if info["grasped"] < 0 else str(info["grasped"])
                print(f"\rPlaced: {info['successes']}/{info['n_objects']}  "
                      f"Grip: {grasp_str}  Slots: [{status}]", end="")

                if step.done:
                    print("\nAll objects placed! Press Triangle to reset.")
