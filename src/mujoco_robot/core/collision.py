"""Self-collision detection for UR robot arms.

Counts MuJoCo contacts between non-adjacent robot links using the
dual-geom architecture (``_col`` suffix = collision geom).

Usage::

    detector = CollisionDetector(model)
    count = detector.count(data)
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Set

import mujoco
import numpy as np


# The six adjacent body pairs whose contacts are always excluded
_ADJACENT_PAIRS: Set[FrozenSet[str]] = {
    frozenset({"base", "shoulder"}),
    frozenset({"shoulder", "upper_arm"}),
    frozenset({"upper_arm", "forearm"}),
    frozenset({"forearm", "wrist1"}),
    frozenset({"wrist1", "wrist2"}),
    frozenset({"wrist2", "wrist3"}),
}

_ROBOT_BODIES = [
    "base", "shoulder", "upper_arm", "forearm",
    "wrist1", "wrist2", "wrist3",
]


class CollisionDetector:
    """Detects self-collision between non-adjacent robot links.

    Parameters
    ----------
    model : mujoco.MjModel
        Compiled model — used to build the geom-to-body mapping.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self._col_geom_to_body: Dict[int, str] = {}
        for gi in range(model.ngeom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi)
            if gname and gname.endswith("_col"):
                body_id = model.geom_bodyid[gi]
                bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if bname in _ROBOT_BODIES:
                    self._col_geom_to_body[gi] = bname

    def count(self, data: mujoco.MjData) -> int:
        """Return the number of non-adjacent self-collision contacts."""
        n = 0
        for ci in range(data.ncon):
            c = data.contact[ci]
            b1 = self._col_geom_to_body.get(c.geom1)
            b2 = self._col_geom_to_body.get(c.geom2)
            if b1 is None or b2 is None:
                continue
            if b1 == b2:
                continue
            if frozenset({b1, b2}) in _ADJACENT_PAIRS:
                continue
            n += 1
        return n
