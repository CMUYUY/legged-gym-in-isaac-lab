# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Package for locomotion tasks ported from legged_gym."""

from .legged_robot_env import LeggedRobotEnv
from .legged_robot_cfg import (
    LeggedRobotEnvCfg,
    AnymalCFlatEnvCfg,
)

__all__ = [
    "LeggedRobotEnv",
    "LeggedRobotEnvCfg",
    "AnymalCFlatEnvCfg",
]
