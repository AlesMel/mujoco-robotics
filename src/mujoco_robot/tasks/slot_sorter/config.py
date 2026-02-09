"""Slot-sorter task configuration objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class SlotSorterTaskConfig:
    """High-level configuration for the slot-sorter task."""

    model_path: str | None = None
    actuator_profile: str = "ur5e"
    time_limit: int = 400
    seed: int | None = None
    render: bool = False
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
