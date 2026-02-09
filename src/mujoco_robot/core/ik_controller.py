"""Damped-least-squares IK controller for 6-DOF UR arms.

Computes joint velocity commands that drive the end-effector toward a
Cartesian pose target (position + full orientation) using the analytic
Jacobian pseudo-inverse.

Usage::

    ik = IKController(model, data, ee_site_id, robot_dofs, damping=0.02)
    qvel = ik.solve(target_pos, target_quat)
"""
from __future__ import annotations

import mujoco
import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Quaternion math helpers  (numpy, wxyz convention like MuJoCo)
# ─────────────────────────────────────────────────────────────────────

def _mat_to_quat(mat3x3: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a unit quaternion (w,x,y,z).

    Uses Shepperd's method for numerical stability.
    """
    m = mat3x3
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit quats): (w, -x, -y, -z)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (w,x,y,z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_unique(q: np.ndarray) -> np.ndarray:
    """Ensure w ≥ 0 (resolve q / -q ambiguity)."""
    return -q if q[0] < 0 else q.copy()


def quat_to_rot6d(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion (w,x,y,z) to the 6-D rotation representation.

    Returns the first two columns of the rotation matrix, flattened to
    a 6-D vector.  This is the minimal *continuous* representation of
    SO(3) — see Zhou et al., "On the Continuity of Rotation
    Representations in Neural Networks" (CVPR 2019).

    Unlike quaternions, this representation has **no discontinuities**
    (no double-cover / sign-flip problem), which makes it much easier
    for neural networks to learn from.
    """
    w, x, y, z = q
    # First column of rotation matrix
    r1 = np.array([
        1 - 2*(y*y + z*z),
        2*(x*y + w*z),
        2*(x*z - w*y),
    ])
    # Second column of rotation matrix
    r2 = np.array([
        2*(x*y - w*z),
        1 - 2*(x*x + z*z),
        2*(y*z + w*x),
    ])
    return np.concatenate([r1, r2])


def rot_mat_to_rot6d(mat3x3: np.ndarray) -> np.ndarray:
    """Extract the 6-D continuous rotation from a 3×3 rotation matrix.

    Returns the first two columns flattened: ``[r00,r10,r20, r01,r11,r21]``.
    """
    return np.concatenate([mat3x3[:, 0], mat3x3[:, 1]])


def axis_angle_from_quat(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion (w,x,y,z) to axis-angle (3-D) vector.

    The vector direction is the rotation axis, its norm is the angle
    in radians ∈ [0, π].  Matches Isaac Lab's implementation.
    """
    q = quat_unique(q)
    sin_half = np.linalg.norm(q[1:4])
    if sin_half < 1e-10:
        return np.zeros(3)
    half_angle = np.arctan2(sin_half, q[0])
    axis = q[1:4] / sin_half
    return axis * (2.0 * half_angle)


def quat_error_magnitude(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angular error between two quaternions in radians ∈ [0, π].

    Equivalent to Isaac Lab's ``quat_error_magnitude``:
    ``‖axis_angle_from_quat(q_err)‖``
    """
    q_err = quat_multiply(q1, quat_conjugate(q2))
    return float(np.linalg.norm(axis_angle_from_quat(q_err)))


def orientation_error_axis_angle(
    current_quat: np.ndarray,
    target_quat: np.ndarray,
) -> np.ndarray:
    """Compute the orientation error as a 3-D axis-angle vector.

    The returned vector points from *current* toward *target*;
    its norm is the angular error in radians.
    """
    q_err = quat_multiply(target_quat, quat_conjugate(current_quat))
    return axis_angle_from_quat(q_err)


class IKController:
    """Damped-least-squares Cartesian IK for a 6-DOF arm.

    Drives the end-effector toward a full 6-DOF pose target
    (position + orientation quaternion) using all 6 rows of the
    site Jacobian (3 translational + 3 rotational).

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
    position_weight : float
        Multiplier on position error relative to orientation error
        in the IK target vector.  A value > 1 makes the solver
        prioritise reaching the target position over matching
        orientation.  Default 5.0 (position is 5× more important).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ee_site: int,
        robot_dofs: list[int],
        damping: float = 0.02,
        position_weight: float = 5.0,
    ) -> None:
        self.model = model
        self.data = data
        self.ee_site = ee_site
        self.robot_dofs = robot_dofs
        self.damping = damping
        self.position_weight = position_weight

    # ------------------------------------------------------------------ API
    def ee_position(self) -> np.ndarray:
        """Current EE position (3-D)."""
        return self.data.site_xpos[self.ee_site].copy()

    def ee_quat(self) -> np.ndarray:
        """Current EE orientation as unit quaternion (w,x,y,z).

        Computed from the site's 3×3 rotation matrix.
        """
        mat = self.data.site_xmat[self.ee_site].reshape(3, 3)
        return _mat_to_quat(mat)

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
    ) -> np.ndarray:
        """Compute joint-velocity command toward a full 6-DOF target.

        Parameters
        ----------
        target_pos : (3,) array
            Desired end-effector world position.
        target_quat : (4,) array
            Desired end-effector orientation as unit quaternion (w,x,y,z).

        Returns
        -------
        qvel : (n_joints,) array
            Joint-velocity command.
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site)

        pos_err = target_pos - self.data.site_xpos[self.ee_site]
        ori_err = orientation_error_axis_angle(self.ee_quat(), target_quat)

        # Clamp orientation error magnitude so it doesn't dominate the
        # IK solve and produce wild wrist velocities.  Large orientation
        # errors (>1 rad) produce huge rotational demands that drown
        # out position tracking in the 6×6 system.  Clamping to
        # ``max_ori_err`` lets the solver make incremental progress on
        # orientation without sacrificing position accuracy.
        max_ori_err = 0.5  # radians — max orientation error per solve
        ori_mag = np.linalg.norm(ori_err)
        if ori_mag > max_ori_err:
            ori_err = ori_err * (max_ori_err / ori_mag)

        # Weight position error so the solver prioritises position
        # tracking over orientation when both are far from target.
        w = self.position_weight
        cols = self.robot_dofs
        J_pos = jacp[:, cols] * w        # (3, n_joints)
        J_rot = jacr[:, cols]            # (3, n_joints)
        J = np.vstack([J_pos, J_rot])    # (6, n_joints)

        target_vec = np.concatenate([pos_err * w, ori_err])  # (6,)

        lam = self.damping
        JJT = J @ J.T + (lam ** 2) * np.eye(6)
        return J.T @ np.linalg.solve(JJT, target_vec)
