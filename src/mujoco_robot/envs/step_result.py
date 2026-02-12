"""Shared step-result container for all task environments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class StepResult:
    """Container returned by ``env.step()`` for non-Gymnasium usage.

    All task environments (reach, lift-suction, slot-sorter) share this
    single definition instead of duplicating the dataclass.
    """

    obs: np.ndarray
    reward: float
    done: bool
    info: Dict
