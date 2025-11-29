# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Task registration for legged_gym environments."""

import gymnasium as gym

from legged_gym_isaaclab.tasks.locomotion import (
    LeggedRobotEnv,
    AnymalCFlatEnvCfg,
)

##
# Register Gym environments
##

gym.register(
    id="Isaac-Legged-Anymal-C-Flat-v0",
    entry_point="legged_gym_isaaclab.tasks.locomotion:LeggedRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCFlatEnvCfg,
        "rsl_rl_cfg_entry_point": None,  # Use default RSL-RL config
    },
)
