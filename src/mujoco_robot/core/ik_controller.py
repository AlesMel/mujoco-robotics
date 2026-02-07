"""Damped-least-squares IK controller for 6-DOF UR arms.

Computes joint velocity commands that drive the end-effector toward a
Cartesian + yaw target using the analytic Jacobian pseudo-inverse.

Usage::

    ik = IKController(model, data, ee_site_id, robot_dofs, damping=0.02)
    qvel = ik.solve(target_pos, target_yaw)
"""
from __future__ import annotations

import math

import mujoco
import numpy as np


class IKController:
    """Damped-least-squares Cartesian IK for a 6-DOF arm.

    Parameters
    ----------
    model : mujoco.MjModel
        Compiled MuJoCo model.
    data : mujoco.MjData
        Simulation data (updated externally via ``mj_step``).
    ee_site : int
        MuJoCo site ID for the end-effector.
    robot_dofs : list[int]
        Indices into ``model.nv`` for the robot joints.
    damping : float
        Damping factor for the pseudo-inverse (``lambda``).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ee_site: int,
        robot_dofs: list[int],
        damping: float = 0.02,
    ) -> None:
        self.model = model
        self.data = data
        self.ee_site = ee_site
        self.robot_dofs = robot_dofs
        self.damping = damping

    # ------------------------------------------------------------------ API
    def ee_position(self) -> np.ndarray:
        """Current EE position (3-D)."""
        return self.data.site_xpos[self.ee_site].copy()

    def ee_yaw(self) -> float:
        """Current EE yaw angle (radians)."""
        mat = self.data.site_xmat[self.ee_site].reshape(3, 3)
        return math.atan2(mat[1, 0], mat[0, 0])

    def solve(self, target_pos: np.ndarray, target_yaw: float) -> np.ndarray:
        """Compute joint-velocity command toward the target.

        Parameters
        ----------
        target_pos : (3,) array
            Desired end-effector world position.
        target_yaw : float
            Desired end-effector yaw (radians).

        Returns
        -------
        qvel : (n_joints,) array
            Joint-velocity command.
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site)

        pos_err = target_pos - self.data.site_xpos[self.ee_site]
        yaw_err = target_yaw - self.ee_yaw()
        yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi

        target_vec = np.concatenate([pos_err, [yaw_err]])
        cols = self.robot_dofs
        J = np.vstack([jacp[:, cols], jacr[2:3, cols]])  # (4, n_joints)

        lam = self.damping
        JJT = J @ J.T + (lam ** 2) * np.eye(4)
        return J.T @ np.linalg.solve(JJT, target_vec)
