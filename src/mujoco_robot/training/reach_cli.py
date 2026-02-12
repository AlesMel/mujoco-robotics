"""Shared CLI helpers for reach training entrypoints."""
from __future__ import annotations

import argparse
from typing import Any, Sequence


DEFAULT_CFG_NAME = "ur3e_joint_pos_dense_stable"


def add_reach_train_args(
    parser: argparse.ArgumentParser,
    *,
    default_total_timesteps: int = 500_000,
    default_n_envs: int = 16,
    control_variant_choices: Sequence[str] | None = None,
) -> None:
    """Attach reach-training args to an existing parser."""
    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        choices=["ur3e", "ur5e"],
        help="Optional robot override for selected cfg profile.",
    )
    parser.add_argument("--total-timesteps", type=int, default=default_total_timesteps)
    parser.add_argument("--n-envs", type=int, default=default_n_envs)

    cv_kwargs: dict[str, Any] = {}
    if control_variant_choices is not None:
        cv_kwargs["choices"] = sorted(control_variant_choices)
        cv_help = "Control variant. Available: " + ", ".join(sorted(control_variant_choices))
    else:
        cv_help = "Control variant key (joint_pos, ik_rel, ik_abs, ...)."
    parser.add_argument(
        "--control-variant",
        type=str,
        default=None,
        help=cv_help,
        **cv_kwargs,
    )
    parser.add_argument(
        "--reach-threshold",
        type=float,
        default=None,
        help="Optional override for position tolerance (metres).",
    )
    parser.add_argument(
        "--ori-threshold",
        type=float,
        default=None,
        help="Optional override for orientation tolerance (radians).",
    )
    parser.add_argument(
        "--success-hold-steps",
        type=int,
        default=None,
        help="Optional override for hold-success step count.",
    )
    parser.add_argument(
        "--success-bonus",
        type=float,
        default=None,
        help="Optional override for one-time hold-success bonus reward.",
    )
    parser.add_argument(
        "--stay-reward-weight",
        type=float,
        default=None,
        help="Optional override for per-second stay reward weight.",
    )
    parser.add_argument(
        "--resample-on-success",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional override for goal resampling after hold-success.",
    )
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-video-every", type=int, default=50_000)
    parser.add_argument(
        "--cfg-name",
        type=str,
        default=DEFAULT_CFG_NAME,
        help="Reach config profile name (e.g. ur5e_joint_pos, ur3e_ik_rel).",
    )
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Stable-Baselines3 tqdm/rich progress bar.",
    )
    parser.add_argument(
        "--sb3-verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Stable-Baselines3 verbosity (0 recommended with progress bar).",
    )
    parser.add_argument(
        "--callback-new-best-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, callback prints only when eval return reaches a new best.",
    )


def reach_train_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Convert parsed CLI args into kwargs for ``train_reach_ppo``."""
    return {
        "robot": args.robot,
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "save_video": args.save_video,
        "save_video_every": args.save_video_every,
        "control_variant": args.control_variant,
        "reach_threshold": args.reach_threshold,
        "ori_threshold": args.ori_threshold,
        "success_hold_steps": args.success_hold_steps,
        "success_bonus": args.success_bonus,
        "stay_reward_weight": args.stay_reward_weight,
        "resample_on_success": args.resample_on_success,
        "progress_bar": args.progress_bar,
        "sb3_verbose": args.sb3_verbose,
        "callback_new_best_only": args.callback_new_best_only,
        "cfg_name": args.cfg_name,
    }
