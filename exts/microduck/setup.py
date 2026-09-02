"""Setup configuration for the isaaclab-microduck extension."""
from setuptools import setup

setup(
    name="isaaclab-microduck",
    version="0.1.0",
    author="Yoll",
    description="RL training environments for the MicroDuck biped robot in Isaac Lab",
    license="BSD-3-Clause",
    python_requires=">=3.12",
    packages=["isaaclab_microduck", "isaaclab_microduck.robot", "isaaclab_microduck.tasks"],
    package_dir={"": "."},
    install_requires=["isaaclab>=2.0.0"],
    entry_points={
        "isaaclab.tasks": [
            "Isaac-Velocity-Flat-MicroDuck-v0 = isaaclab_microduck.tasks.microduck_flat_env_cfg:MicroDuckFlatEnvCfg",
            "Isaac-Velocity-Flat-MicroDuck-Play-v0 = isaaclab_microduck.tasks.microduck_flat_env_cfg:MicroDuckFlatEnvCfg_PLAY",
        ],
    },
)
