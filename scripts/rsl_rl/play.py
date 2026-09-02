"""Play a trained MicroDuck PPO policy with high-quality video recording.

Loads a trained checkpoint, spawns 64 ducks at duck-level camera angle,
and records a 20-second CRF-10 H.264 video.

Usage::

    python play.py --checkpoint logs/rsl_rl/microduck_flat/<run>/model_4999.pt
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play MicroDuck PPO (kitless Newton).")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="logs/rsl_rl/microduck_flat/2026-09-01_11-57-16/model_4999.pt",
    help="Path to the .pt checkpoint file.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="logs/rsl_rl/microduck_flat/play_videos",
    help="Directory for the output video.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

import math

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_tasks.utils.sim_launcher import launch_simulation

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlBaseRunnerCfg, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner


env_cfg = load_cfg_from_registry("Isaac-Velocity-Flat-MicroDuck-v0", "env_cfg_entry_point")
env_cfg = resolve_presets(env_cfg, selected=("newton_mjwarp",))
agent_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(
    "Isaac-Velocity-Flat-MicroDuck-v0", "rsl_rl_cfg_entry_point"
)

installed_version = metadata.version("rsl-rl-lib")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

# Play config: 64 envs, tight spacing, no randomization
env_cfg.scene.num_envs = 64
env_cfg.scene.env_spacing = 0.8  # tighter packing  # wider spacing between ducks
env_cfg.sim.device = args_cli.device
env_cfg.commands.base_velocity.debug_vis = False
# Ducks face +X and walk STRAIGHT toward the camera placed at +X
env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.75, 0.75)  # forward only
env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)  # spawn facing +X
# Spawn ducks 2m further in -X from grid positions (closer to camera at +X=8)
env_cfg.events.reset_base.params["pose_range"]["x"] = (-4.0, -3.0)  # ducks much closer to camera
env_cfg.observations.policy.enable_corruption = False
env_cfg.events.base_external_force_torque = None
env_cfg.events.push_robot = None
# Force 1920x1080 for sharp recordings
env_cfg.video_recorder.window_width = 1920
env_cfg.video_recorder.window_height = 1080

env_step_dt = env_cfg.sim.dt * env_cfg.decimation

output_dir = Path(args_cli.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = Path(args_cli.checkpoint)
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path.resolve()}")
print(f"[INFO] Loading checkpoint: {checkpoint_path}")

with launch_simulation(env_cfg, args_cli):
    env = gym.make(
        "Isaac-Velocity-Flat-MicroDuck-v0",
        cfg=env_cfg,
        render_mode="rgb_array",
    )

    # Camera: close but far enough to see the walking path, on +X side
    # Ducks walk +X (toward camera at x=6). Camera at (6, -4, 3) looking at (0, 0, 0.2).
    # Ducks walk 0 → 5.6 m over 10 s at 0.75 m/s — plenty of distance.
    # Camera at duck level (Z=0.3), looking straight at them head-on
    # Ducks spawn at X≈-3, walk toward camera at X=8 (11m approach)
    if hasattr(env.unwrapped, "video_recorder") and env.unwrapped.video_recorder._capture:
        env.unwrapped.video_recorder._capture.update_camera(
            position=(3.0, -1.0, 0.3), target=(-8.0, 0.0, 0.3)
        )
        env.unwrapped.video_recorder._capture.cfg.horiz_fov_deg = 60.0
        print("[INFO] Camera close at duck level: eye=(3, -1, 0.3) → target=(-8, 0, 0.3), FOV=60°")
    else:
        print("[WARN] Could not find video_recorder._capture to set camera")

    # Record 10-second video at 30 FPS → 300 frames
    # 20-second video at 30 FPS → 600 frames
    video_length_frames = 20 * 30
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(output_dir / "videos"),
        step_trigger=lambda step: step == 0,
        video_length=video_length_frames,
        fps=30,
        disable_logger=True,
    )

    # Override stop_recording for maximum quality (CRF 10 ≈ near-lossless)
    original_stop = env.stop_recording

    def _ultra_quality_stop():
        if len(env.recorded_frames) == 0:
            print("[WARN] No frames recorded, skipping video.")
            env.recording = False
            env._video_name = None
            return
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        clip = ImageSequenceClip(env.recorded_frames, fps=env.frames_per_sec)
        path = os.path.join(env.video_folder, f"{env._video_name}.mp4")
        clip.write_videofile(
            path,
            logger=None,
            codec="libx264",
            preset="slow",
            threads=4,
            ffmpeg_params=["-crf", "10"],
        )
        del clip
        del env.recorded_frames
        env.recorded_frames = []
        env.recording = False
        env._video_name = None

    env.stop_recording = _ultra_quality_stop
    print(
        f"[INFO] Video: 20 seconds, 30 FPS, libx264 CRF 10 (near-lossless), "
        f"{video_length_frames} frames"
    )

    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env_wrapped.unwrapped.render_enabled = False

    runner = OnPolicyRunner(
        env_wrapped, agent_cfg.to_dict(), log_dir=str(output_dir), device=agent_cfg.device
    )
    runner.load(str(checkpoint_path))
    print("[INFO] Checkpoint loaded. Starting playback.")

    obs, _ = env_wrapped.reset()
    for step in range(video_length_frames):
        with torch.inference_mode():
            actions = runner.alg.act(obs)
        obs, _, _, _ = env_wrapped.step(actions.to(env_wrapped.device))

    print("[INFO] Playback complete. Video saved.")
    env_wrapped.close()
