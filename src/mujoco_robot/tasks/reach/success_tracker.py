"""Tracks success streaks, hold completions, and bonus rewards.

Encapsulates the success/hold bookkeeping that was previously scattered
across ``rewarding.py``, ``resetting.py``, and ``goals.py`` as raw
attribute mutations on the environment object.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SuccessResult:
    """Snapshot of success tracking state after one step update."""

    success: bool = False
    hold_success: bool = False
    first_hold: bool = False
    streak_steps: int = 0
    stay_reward: float = 0.0
    success_bonus: float = 0.0
    total_reward_add: float = 0.0


class SuccessTracker:
    """Manages goal-reaching streaks and hold-success logic.

    Parameters
    ----------
    hold_steps : int
        Number of consecutive success steps before a hold is confirmed.
    bonus : float
        One-time reward bonus on the first confirmed hold.
    stay_weight : float
        Per-second reward for continuing to hold after confirmation.
    """

    def __init__(
        self,
        hold_steps: int = 10,
        bonus: float = 0.25,
        stay_weight: float = 0.05,
    ) -> None:
        self.hold_steps = max(1, int(hold_steps))
        self.bonus = float(bonus)
        self.stay_weight = float(stay_weight)

        # Counters (reset each episode)
        self.goals_reached: int = 0
        self.goals_held: int = 0
        self.streak_steps: int = 0

        # Per-step state
        self._prev_success: bool = False
        self._hold_completed: bool = False

    # ------------------------------------------------------------------ API

    def reset(self) -> None:
        """Reset all counters for a new episode."""
        self.goals_reached = 0
        self.goals_held = 0
        self.streak_steps = 0
        self._prev_success = False
        self._hold_completed = False

    def reset_for_new_goal(self) -> None:
        """Reset streak state when the goal command is resampled."""
        self.streak_steps = 0
        self._hold_completed = False

    def update(self, success: bool, step_dt: float) -> SuccessResult:
        """Update tracking after one step and return the result snapshot.

        Parameters
        ----------
        success : bool
            Whether the agent is within the success thresholds this step.
        step_dt : float
            Wall-clock duration of one control step (seconds).

        Returns
        -------
        SuccessResult
            Holds all derived flags and bonus/stay rewards for the step.
        """
        # First-time reach in a new streak
        if success and not self._prev_success:
            self.goals_reached += 1
        self._prev_success = success

        # Streak counting
        if success:
            self.streak_steps += 1
        else:
            self.streak_steps = 0

        hold_success = bool(success and self.streak_steps >= self.hold_steps)
        first_hold = bool(hold_success and not self._hold_completed)
        if first_hold:
            self._hold_completed = True
            self.goals_held += 1

        stay_reward = float(self.stay_weight * step_dt) if hold_success else 0.0
        bonus = float(self.bonus) if first_hold else 0.0

        return SuccessResult(
            success=success,
            hold_success=hold_success,
            first_hold=first_hold,
            streak_steps=self.streak_steps,
            stay_reward=stay_reward,
            success_bonus=bonus,
            total_reward_add=stay_reward + bonus,
        )
