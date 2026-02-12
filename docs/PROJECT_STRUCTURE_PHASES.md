# Project Structure Refactor Phases

This plan tracks the staged cleanup of the project structure.

## Phase 1 (completed)

- Unify reach runtime entrypoints:
  - `make_task("reach")` and Gym IDs now route through manager-based reach runtime.
- Consolidate train CLI wiring:
  - `src/mujoco_robot/scripts/train.py` now uses shared reach CLI helpers.
  - `scripts/train.py` is now a thin wrapper.
- Remove package-path anti-patterns:
  - replaced `from training.*` imports with `from mujoco_robot.training.*`.
  - removed direct `src/mujoco_robot` path injection hacks in tests/scripts.
- Start cfg-first override policy:
  - reach CLI overrides are optional (`None` by default) so cfg profile values stay authoritative.

## Phase 2 (completed)

- Reduce `reach_env_base.py` responsibilities:
  - extracted HUD rendering logic to `src/mujoco_robot/envs/reach/rendering.py`.
  - extracted goal sampling + command-resample helpers to
    `src/mujoco_robot/envs/reach/goals.py`.
  - extracted reward/info assembly to
    `src/mujoco_robot/envs/reach/rewarding.py`.
  - extracted multi-camera composition helper to
    `src/mujoco_robot/envs/reach/rendering.py`.
  - extracted IK/cartesian helper math to
    `src/mujoco_robot/envs/reach/kinematics.py`.
  - extracted reset/bootstrap flow to
    `src/mujoco_robot/envs/reach/resetting.py`.

## Phase 3 (next)

- Formal deprecation path:
  - compatibility shims now emit deprecation warnings.
  - target removal release: `v0.4.0`.
- Remove remaining deprecated import paths after one release cycle.

## Phase 4 (next)

- CI quality gates:
  - added baseline GitHub Actions workflow for compile + core tests.
- Expand CI matrix:
  - full pytest suite
  - lint/format checks
  - optional smoke evaluation job.
