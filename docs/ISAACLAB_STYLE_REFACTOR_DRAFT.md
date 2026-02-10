# IsaacLab-Style Reach Refactor Draft

This document proposes a project structure and migration plan to make the reach stack feel closer to IsaacLab: config-first, manager-based, and cleanly separated logic.

## Status

Phase 1 scaffold is now implemented:
1. manager-based runtime skeleton (`envs/manager_based/*`)
2. config-first reach package scaffold (`tasks/manager_based/manipulation/reach/*`)
3. profile registry and cfg adapters to current reach runtime
4. compatibility task entry in task registry (`reach_manager_based`)
5. training entrypoint wiring (`scripts/train.py --task reach --cfg-name ...`)

Phase 2 migration is now implemented for reach MDP logic:
1. `tasks/manager_based/manipulation/reach/mdp/*` now contains real term/config/manager logic
2. reach runtime imports now point to the new task-path MDP modules
3. legacy `envs/reach/mdp/*` is preserved as compatibility shims

Phase 3 runtime extraction is now implemented:
1. generic `ManagerRuntime` is used by reach to host action/command/observation/reward/termination managers
2. `reach_manager_based` now builds a runtime-backed env directly (not via `URReachEnv` factory adapter)
3. legacy env APIs remain for compatibility while the new manager-based task path is the primary implementation path

Phase 4 entrypoint consolidation is now implemented:
1. reach training CLI args/defaults are centralized in `training/reach_cli.py`
2. both `scripts/train.py` and `training/train_reach.py` consume the shared reach CLI wiring
3. cfg/profile-first selection is standardized via `--cfg-name`

Phase 5 deprecation cleanup is now implemented (breaking cleanup):
1. removed legacy `mujoco_robot.envs.reach_env` shim and `action_mode` alias path
2. removed deprecated reach task/runtime aliases from training CLI
3. removed compatibility `envs/reach/mdp/*` shim package
4. `reach` task now points directly to manager-based runtime/config

## 1. Objectives

1. Make environment construction config-first instead of kwargs-heavy.
2. Keep task logic modular: actions, observations, rewards, terminations, commands, events, curriculum.
3. Separate generic manager runtime from task-specific definitions.
4. Keep training entrypoints thin (select cfg/profile, run trainer).
5. Preserve backward compatibility during migration.

## 2. Non-Goals (for phase 1)

1. No immediate rewrite of slot sorter internals.
2. No behavior change to reward/termination dynamics unless explicitly chosen.
3. No immediate deletion of current env IDs or public APIs.

## 3. Target Package Layout

```text
src/mujoco_robot/
  envs/
    manager_based/
      base_env.py                  # generic manager-based env runtime
      base_rl_env.py               # RL-facing wrapper runtime (step/reset API glue)
      manager_runtime.py           # manager orchestration utilities
  tasks/
    manager_based/
      manipulation/
        reach/
          __init__.py              # registration + exports
          reach_env.py             # ReachManagerBasedRLEnv class
          reach_env_cfg.py         # top-level env cfg composition
          mdp/
            actions.py
            observations.py
            rewards.py
            terminations.py
            commands.py
            events.py
            curriculum.py
            schemas.py             # term cfg dataclasses if needed
          config/
            ur3e/
              joint_pos_env_cfg.py
              ik_rel_env_cfg.py
              ik_abs_env_cfg.py
            ur5e/
              joint_pos_env_cfg.py
              ik_rel_env_cfg.py
              ik_abs_env_cfg.py
```

Current `src/mujoco_robot/envs/reach/*` remains in place during migration and forwards to new implementation once stable.

## 4. Config Hierarchy (Single Source of Truth)

Phase-2 target:

1. `ReachEnvCfg`
2. `SceneCfg`
3. `SimulationCfg`
4. `EpisodeCfg`
5. `ActionCfg`
6. `ObservationCfg`
7. `CommandCfg`
8. `RewardCfg`
9. `TerminationCfg`
10. `EventCfg`
11. `CurriculumCfg`

Important principle:
1. All defaults live in cfg objects.
2. Runtime classes read cfg only.
3. CLI/training only selects cfg/profile + optional small overrides.

## 5. Profile Model

Profiles are composed from base cfg:

1. `reach_base_cfg`
2. `ur3e_overlay_cfg` / `ur5e_overlay_cfg`
3. `control_overlay_cfg` (`joint_pos`, `ik_rel`, `ik_abs`)
4. `reward_profile_cfg` (`dense_stable`, `dense_resample`, `isaac_compat`, `sparse_hold`)

Composition output is one final `ReachEnvCfg`.

## 6. API Shape (Post-Migration)

Raw env:

```python
from mujoco_robot.tasks.manager_based.manipulation.reach import ReachManagerBasedRLEnv
from mujoco_robot.tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

cfg = ReachEnvCfg()
env = ReachManagerBasedRLEnv(cfg=cfg)
```

Gymnasium entrypoint:

```python
import gymnasium
env = gymnasium.make("MuJoCoRobot/Reach-v0", cfg_name="ur3e_joint_pos_dense_stable")
```

Legacy kwargs API remains, internally translated to cfg with deprecation warning.

## 7. Migration Plan

### Phase 0: Baseline Lock

1. Freeze current behavior with tests for:
2. reset/step contract
3. reward decomposition terms
4. termination flags
5. goal resampling logic
6. training smoke path

Exit criteria:
1. We can detect behavior regressions quickly.

### Phase 1: Scaffolding (No Behavior Change)

1. Add target folders and placeholder modules.
2. Add cfg dataclasses mirroring current defaults.
3. Add adapter from cfg to current `URReachEnvBase` constructor.

Exit criteria:
1. New cfg objects can instantiate current env behavior exactly.

### Phase 2: Move Reach Term Logic

1. Move reach term logic fully into new `tasks/.../reach/mdp/`.
2. Keep runtime unchanged, swap imports progressively.

Exit criteria:
1. Term logic no longer depends on old path layout.

### Phase 3: Manager Runtime Extraction

1. Introduce generic manager runtime in `envs/manager_based`.
2. Port reach to this runtime.
3. Keep old `envs/reach/*` as wrappers.

Exit criteria:
1. Reach works end-to-end using new manager runtime.

### Phase 4: Entry Point Consolidation

1. Make training scripts profile/cfg-driven.
2. Keep one primary training entrypoint, keep others thin wrappers.

Exit criteria:
1. No duplicated default definitions across scripts/env files.

### Phase 5: Deprecation Cleanup

1. Emit deprecation warnings for legacy kwargs and old module imports.
2. Remove legacy path after one release window.

## 8. Mapping from Current Files

1. `src/mujoco_robot/envs/reach/reach_env_base.py` -> split into:
2. generic runtime (`envs/manager_based/*`)
3. reach task runtime (`tasks/.../reach/reach_env.py`)
4. cfg (`tasks/.../reach/reach_env_cfg.py`)

5. `src/mujoco_robot/envs/reach/mdp/*` -> move to:
6. `src/mujoco_robot/tasks/manager_based/manipulation/reach/mdp/*`

7. `src/mujoco_robot/tasks/reach/*` -> removed in phase-5 cleanup.

## 9. Compatibility Strategy

1. Keep existing gym IDs (`MuJoCoRobot/Reach-*`) unchanged.
2. Legacy `URReachEnv`/`ReachGymnasium` imports are removed in the breaking cleanup.
3. Keep API surface centered on manager-based cfg profiles and explicit reach variants.

## 10. First PR Scope (Recommended)

1. Add new folder skeleton and cfg classes.
2. Add cfg-to-legacy adapter.
3. Add one profile file (`ur3e_joint_pos_dense_stable`).
4. Add docs for selecting cfg/profile in training.

No runtime behavior change in first PR.

## 11. Open Decisions

1. Keep both old and new paths for one release cycle, or hard-cut immediately?
2. Keep both training entrypoints, or consolidate to one now?
3. Keep reward curriculum in default profile, or only in advanced profiles?
4. Keep `isaac_compat` profile exact even if defaults differ from your current best?
5. Require cfg-only in new code (no direct kwargs), or allow mixed style temporarily?
