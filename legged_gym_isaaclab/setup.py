"""Installation script for the legged_gym_isaaclab extension."""

from setuptools import setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "numpy",
    "torch",
]

# Installation operation
setup(
    name="legged_gym_isaaclab",
    author="Legged Gym Team",
    version="1.0.0",
    description="Legged Gym environments ported to Isaac Lab",
    keywords=["robotics", "rl", "isaac lab"],
    install_requires=INSTALL_REQUIRES,
    packages=["legged_gym_isaaclab"],
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.10"],
    zip_safe=False,
)
