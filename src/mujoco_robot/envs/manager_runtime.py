"""Generic runtime container for manager-based environments."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ManagerRuntime:
    """Container for named manager objects used by one environment."""

    managers: Dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, manager: Any) -> None:
        self.managers[name] = manager

    def add_many(self, **named_managers: Any) -> None:
        for name, manager in named_managers.items():
            self.add(name, manager)

    def get(self, name: str) -> Any:
        return self.managers[name]

    def has(self, name: str) -> bool:
        return name in self.managers

    def require(self, name: str) -> Any:
        if name not in self.managers:
            available = ", ".join(sorted(self.managers))
            raise KeyError(
                f"Manager '{name}' is not registered. Available: [{available}]"
            )
        return self.managers[name]
