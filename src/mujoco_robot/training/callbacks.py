"""Stable-Baselines3 callbacks for periodic video recording.

Usage::

    from mujoco_robot.training import BestEpisodeVideoCallback

    video_cb = BestEpisodeVideoCallback(
        make_eval_env=lambda: Monitor(make_reach_manager_based_gymnasium("ur3e_joint_pos")),
        save_every_timesteps=50_000,
        video_dir="videos",
        env_name="reach_ur5e",
    )
    model.learn(total_timesteps=500_000, callback=[video_cb])
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import gymnasium
import imageio.v3 as iio
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class CableRoutingCurriculumCallback(BaseCallback):
    """Single-run curriculum for cable routing (easy -> mid -> full)."""

    def __init__(
        self,
        total_timesteps: int,
        stage1_frac: float = 0.35,
        stage2_frac: float = 0.70,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.total_timesteps = int(max(1, total_timesteps))
        self.stage1_frac = float(np.clip(stage1_frac, 0.05, 0.95))
        self.stage2_frac = float(np.clip(stage2_frac, self.stage1_frac + 0.05, 0.99))
        self._active_stage = -1

    def _stage_from_progress(self) -> int:
        progress = float(self.num_timesteps / self.total_timesteps)
        if progress < self.stage1_frac:
            return 0
        if progress < self.stage2_frac:
            return 1
        return 2

    def _apply_stage(self, stage: int) -> None:
        stages = self.training_env.env_method("apply_curriculum_stage", int(stage))
        self._active_stage = int(stage)
        self.logger.record("curriculum/stage", float(self._active_stage))
        if self.verbose:
            uniq = sorted(set(int(s) for s in stages))
            print(
                f"[curriculum] step={self.num_timesteps} "
                f"progress={self.num_timesteps/self.total_timesteps:.2%} "
                f"stage={self._active_stage} env_stages={uniq}"
            )

    def _init_callback(self) -> None:
        self._apply_stage(self._stage_from_progress())

    def _on_step(self) -> bool:
        target_stage = self._stage_from_progress()
        if target_stage != self._active_stage:
            self._apply_stage(target_stage)
        return True


class BestEpisodeVideoCallback(BaseCallback):
    """Records a video of the best-return eval episode every *N* training episodes.

    Videos are saved into ``<video_dir>/<env_name>/<timestamp>/`` so that
    different environments and training runs each get their own subfolder.

    Parameters
    ----------
    make_eval_env : callable
        Zero-arg factory that returns a Gymnasium env with rendering.
    save_every_timesteps : int
        Training timesteps between video recordings.
    video_dir : str
        Root directory for video output.
    env_name : str
        Sub-folder name (e.g. ``"reach_ur5e"``).
    deterministic : bool
        Whether to use deterministic policy during eval.
    vec_norm : VecNormalize | None
        Training VecNormalize whose obs/ret statistics are copied for eval.
    verbose : int
        Verbosity level.
    log_new_best_only : bool
        If True, only print when a new best return is found.
    """

    def __init__(
        self,
        make_eval_env: Callable[[], gymnasium.Env],
        save_every_timesteps: int,
        video_dir: str = "videos",
        env_name: str = "default",
        deterministic: bool = True,
        vec_norm: VecNormalize | None = None,
        verbose: int = 1,
        log_new_best_only: bool = True,
    ):
        super().__init__(verbose)
        self.make_eval_env = make_eval_env
        self.save_every_timesteps = max(1, save_every_timesteps)
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_dir = Path(video_dir) / env_name / run_stamp
        self.deterministic = deterministic
        self.vec_norm = vec_norm
        self.log_new_best_only = bool(log_new_best_only)
        self.best_return = -np.inf
        self.next_record = self.save_every_timesteps

    def _init_callback(self) -> None:
        self.video_dir.mkdir(parents=True, exist_ok=True)

    def _write_video(self, path: Path, frames: list[np.ndarray], fps: int) -> None:
        if not frames:
            return
        frame_list = list(frames)
        if len(frame_list) == 1:
            frame_list.append(frame_list[0].copy())
        try:
            iio.imwrite(
                path,
                frame_list,
                fps=fps,
                codec="libx264",
                pixelformat="yuv420p",
                ffmpeg_log_level="error",
            )
        except TypeError:
            iio.imwrite(path, frame_list, fps=fps)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_record:
            self._record_eval_episode()
            self.next_record += self.save_every_timesteps
        return True

    def _record_eval_episode(self) -> None:
        if self.vec_norm is not None:
            raw_eval = DummyVecEnv([self.make_eval_env])
            env: gymnasium.Env = VecNormalize(
                raw_eval,
                training=False,
                norm_obs=self.vec_norm.norm_obs,
                norm_reward=False,
                clip_obs=self.vec_norm.clip_obs,
            )
            env.obs_rms = self.vec_norm.obs_rms.copy()
            if self.vec_norm.ret_rms is not None:
                env.ret_rms = self.vec_norm.ret_rms.copy()
        else:
            env = self.make_eval_env()

        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        frames = []
        ep_return = 0.0
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            step_out = env.step(action)

            if len(step_out) == 5:
                obs, reward, terminated, truncated, _ = step_out
                if isinstance(reward, np.ndarray):
                    ep_return += float(reward[0])
                    done = bool(terminated[0] or truncated[0])
                else:
                    ep_return += reward
                    done = bool(terminated or truncated)
            else:
                obs, reward, dones, _infos = step_out
                if isinstance(reward, np.ndarray):
                    ep_return += float(reward[0])
                    done = bool(dones[0])
                else:
                    ep_return += reward
                    done = bool(dones)

            frame = env.render()
            if frame is not None:
                frames.append(frame)
        env.close()

        # Derive FPS from env control period
        control_dt = 0.01
        try:
            base_env = None
            if hasattr(env, "venv") and hasattr(env.venv, "envs"):
                base_env = env.venv.envs[0]
            elif hasattr(env, "envs"):
                base_env = env.envs[0]
            if base_env is not None and hasattr(base_env, "env"):
                base_env = base_env.env
            if base_env is not None and hasattr(base_env, "base"):
                ue = base_env.base
                control_dt = float(ue.model.opt.timestep) * float(ue.n_substeps)
        except Exception:
            pass
        fps = max(1, int(round(1.0 / control_dt)))

        ts = int(time.time())
        fname = self.video_dir / f"eval_step_{self.num_timesteps:09d}_{ts}.mp4"
        if frames:
            self._write_video(fname, frames, fps)
            if self.verbose and not self.log_new_best_only:
                print(f"[video] saved eval episode to {fname} "
                      f"(return {ep_return:.3f}, fps={fps})")

        if ep_return > self.best_return and frames:
            self.best_return = ep_return
            best_fname = self.video_dir / "best_episode_latest.mp4"
            self._write_video(best_fname, frames, fps)
            self.logger.record("eval/best_return", float(self.best_return))
            self.logger.record("eval/best_video", str(best_fname))
            if self.verbose:
                print(f"[video] new best return {ep_return:.3f}, "
                      f"updated {best_fname} (fps={fps})")


class ReachEvalMetricsCallback(BaseCallback):
    """Periodic deterministic eval for reach metrics independent of video cadence."""

    def __init__(
        self,
        make_eval_env: Callable[[], gymnasium.Env],
        eval_every_timesteps: int,
        n_eval_episodes: int = 3,
        deterministic: bool = True,
        vec_norm: VecNormalize | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.make_eval_env = make_eval_env
        self.eval_every_timesteps = max(1, int(eval_every_timesteps))
        self.n_eval_episodes = max(1, int(n_eval_episodes))
        self.deterministic = bool(deterministic)
        self.vec_norm = vec_norm
        self.next_eval = self.eval_every_timesteps

    def _on_step(self) -> bool:
        while self.num_timesteps >= self.next_eval:
            self._run_eval()
            self.next_eval += self.eval_every_timesteps
        return True

    def _build_eval_env(self):
        if self.vec_norm is None:
            return self.make_eval_env()

        raw_eval = DummyVecEnv([self.make_eval_env])
        env = VecNormalize(
            raw_eval,
            training=False,
            norm_obs=self.vec_norm.norm_obs,
            norm_reward=False,
            clip_obs=self.vec_norm.clip_obs,
        )
        env.obs_rms = self.vec_norm.obs_rms.copy()
        if self.vec_norm.ret_rms is not None:
            env.ret_rms = self.vec_norm.ret_rms.copy()
        return env

    def _run_eval(self) -> None:
        env = self._build_eval_env()
        returns: list[float] = []
        successes: list[float] = []
        final_dists: list[float] = []
        final_ori_errs: list[float] = []

        try:
            for _ in range(self.n_eval_episodes):
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]

                done = False
                ep_return = 0.0
                last_info: dict | None = None

                while not done:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    step_out = env.step(action)

                    if len(step_out) == 5:
                        obs, reward, terminated, truncated, info = step_out
                        if isinstance(reward, np.ndarray):
                            ep_return += float(reward[0])
                            done = bool(terminated[0] or truncated[0])
                            last_info = info[0] if isinstance(info, (list, tuple)) else info
                        else:
                            ep_return += float(reward)
                            done = bool(terminated or truncated)
                            last_info = info
                    else:
                        obs, reward, dones, infos = step_out
                        if isinstance(reward, np.ndarray):
                            ep_return += float(reward[0])
                            done = bool(dones[0])
                            last_info = infos[0] if isinstance(infos, (list, tuple)) else infos
                        else:
                            ep_return += float(reward)
                            done = bool(dones)
                            last_info = infos

                info_dict = last_info if isinstance(last_info, dict) else {}
                returns.append(float(ep_return))
                hold_success = bool(
                    info_dict.get("hold_success", info_dict.get("success", False))
                )
                successes.append(float(hold_success))
                final_dists.append(float(info_dict.get("dist", np.nan)))
                final_ori_errs.append(float(info_dict.get("ori_err", np.nan)))
        finally:
            env.close()

        self.logger.record("eval/success_rate", float(np.mean(successes)))
        self.logger.record("eval/mean_return", float(np.mean(returns)))
        self.logger.record("eval/mean_dist", float(np.nanmean(final_dists)))
        self.logger.record("eval/mean_ori_err", float(np.nanmean(final_ori_errs)))

        if self.verbose:
            print(
                "[eval] "
                f"step={self.num_timesteps} "
                f"success_rate={np.mean(successes):.3f} "
                f"mean_return={np.mean(returns):.3f} "
                f"mean_dist={np.nanmean(final_dists):.4f} "
                f"mean_ori={np.nanmean(final_ori_errs):.4f}"
            )
