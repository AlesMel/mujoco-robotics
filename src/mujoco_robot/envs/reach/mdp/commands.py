"""Command terms for the reach task."""
from __future__ import annotations

import numpy as np


def uniform_pose_command(env) -> np.ndarray:
    """Sample a pose command ``[x, y, z, qw, qx, qy, qz]`` in world frame."""
    pos = env._sample_goal()
    quat = env._sample_goal_quat()
    return np.concatenate([pos.astype(np.float32), quat.astype(np.float32)])
