# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an RSL-RL PPO agent on the MicroDuck flat-locomotion task (kitless Newton).

Features:
  * All ducks spawn facing +X (heading = 0) with slightly randomised positions.
  * Forward-only velocity commands so the policy learns to walk forward.
  * Headless by default (no Newton GL window); use ``--viz newton`` for an
    interactive viewer.
  * Video clip (5 s) recorded every 500 RL iterations via the gym RecordVideo
    wrapper — no visible window needed.

Usage::

    # Headless training with video: 512 envs, 5000 iters
    python train.py --task Isaac-Velocity-Flat-MicroDuck-v0 --num_envs 512 --max_iterations 5000 --headless --video

    # Quick smoke test
    python train.py --task Isaac-Velocity-Flat-MicroDuck-v0 --num_envs 64 --max_iterations 5 --headless
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train MicroDuck PPO (kitless Newton).")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--max_iterations", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Flat-MicroDuck-v0",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record 5-second video clips every 500 RL iterations.",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=None,
    help="Override video interval (steps).",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help="Override video length (frames).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402 -- trigger gym registration
from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab_tasks.utils.sim_launcher import launch_simulation  # noqa: E402

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from rsl_rl.runners import OnPolicyRunner

# ---------------------------------------------------------------------------
# 1. Load env config (raw, with PresetCfg wrappers) and resolve to Newton
# ---------------------------------------------------------------------------
env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg = resolve_presets(env_cfg, selected=("newton_mjwarp",))

# Override for training
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.scene.env_spacing = 4.0
env_cfg.sim.device = args_cli.device

# Hide velocity command arrows
env_cfg.commands.base_velocity.debug_vis = False

# All ducks face +X (heading=0) at slightly randomised positions
env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)  # forward only
env_cfg.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)  # minimal lateral
env_cfg.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)  # minimal yaw rate
env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)  # spawn facing +X

# Load agent config and migrate deprecated fields for installed RSL-RL
installed_version = metadata.version("rsl-rl-lib")
agent_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
if args_cli.max_iterations is not None:
    agent_cfg.max_iterations = args_cli.max_iterations
agent_cfg.seed = args_cli.seed

# Video: every 500 RL iterations, 5-second clips
env_step_dt = env_cfg.sim.dt * env_cfg.decimation
if args_cli.video:
    if args_cli.video_interval is None:
        args_cli.video_interval = 500 * agent_cfg.num_steps_per_env  # 500 iters → steps
    if args_cli.video_length is None:
        args_cli.video_length = int(5 / env_step_dt)  # 5 seconds → frames

log_root = Path.cwd() / "logs" / "rsl_rl" / agent_cfg.experiment_name
log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir.mkdir(parents=True, exist_ok=True)
env_cfg.log_dir = str(log_dir)

# ---------------------------------------------------------------------------
# 2. Launch simulation (kitless Newton — no AppLauncher/Kit needed)
# ---------------------------------------------------------------------------
with launch_simulation(env_cfg, args_cli):
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(log_dir / "videos"),
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            fps=30,
            disable_logger=True,
        )
        # Override stop_recording for higher-quality H.264 encoding (CRF 18 ≈ visually lossless)
        def _high_quality_stop():
            if len(video_env.recorded_frames) == 0:
                print("[WARN] No frames recorded, skipping video.")
                video_env.recording = False
                video_env._video_name = None
                return
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            clip = ImageSequenceClip(video_env.recorded_frames, fps=video_env.frames_per_sec)
            path = os.path.join(video_env.video_folder, f"{video_env._video_name}.mp4")
            clip.write_videofile(
                path,
                logger=None,
                codec="libx264",
                preset="slow",
                threads=4,
                ffmpeg_params=["-crf", "18"],
            )
            del clip
            del video_env.recorded_frames
            video_env.recorded_frames = []
            video_env.recording = False
            video_env._video_name = None
        video_env.stop_recording = _high_quality_stop
        env = video_env
        print(f"[INFO] Video: every {args_cli.video_interval} steps "
              f"({args_cli.video_length} frames = {args_cli.video_length * env_step_dt:.1f}s, "
              f"30 FPS, libx264 CRF 18)")

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env.unwrapped.render_enabled = False

    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device
    )
    runner.add_git_repo_to_log(__file__)

    print(f"[INFO] Starting PPO training: {args_cli.num_envs} envs, "
          f"{agent_cfg.max_iterations} iterations, headless={not args_cli.visualizer}")
    start = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations)
    print(f"[INFO] Training complete in {round(time.time() - start, 1)} s")
    env.close()
