# IsaacLab MicroDuck

RL training environments for the [MicroDuck](https://github.com/pollen-robotics/microduck) biped robot in **Isaac Lab** with **kitless Newton** physics (MuJoCo Warp backend).

Built on the `ManagerBasedRLEnv` workflow — full observation, reward, action, termination, and event managers.

## Installation

```bash
# 1. Isaac Lab is required (with Newton kitless support)
# Follow: https://isaac-sim.github.io/IsaacLab/main/

# 2. Install the MicroDuck extension
cd exts/microduck
python -m pip install -e .
```

## Training

```bash
# Headless Newton training — 512 envs, 5000 iterations, 10s video every 500
ISAACLAB_PATH=/path/to/isaaclab python scripts/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-MicroDuck-v0 \
    --num_envs 512 \
    --max_iterations 5000 \
    --headless \
    --video

# Quick smoke test
ISAACLAB_PATH=/path/to/isaaclab python scripts/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-MicroDuck-v0 \
    --num_envs 64 \
    --max_iterations 5 \
    --headless
```

## Play (inference + video)

```bash
# Watch a trained policy with a high-quality 10-second video
ISAACLAB_PATH=/path/to/isaaclab python scripts/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-MicroDuck-v0 \
    --checkpoint logs/rsl_rl/microduck_flat/<run>/model_4999.pt \
    --num_envs 64
```

## Tasks

| Task ID | Description |
|---------|-------------|
| `Isaac-Velocity-Flat-MicroDuck-v0` | Forward velocity tracking on flat terrain |
| `Isaac-Velocity-Flat-MicroDuck-Play-v0` | Play variant with fixed commands |

## License

BSD-3-Clause. USD model files licensed CC-BY-NC.
