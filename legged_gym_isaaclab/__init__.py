# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Legged Gym tasks adapted for Isaac Lab.

This package provides Isaac Lab-compatible environments that reproduce
the original legged_gym behaviors using Isaac Lab's APIs.
"""

__version__ = "1.0.0"

##
# Register Gym environments
##

# Import will register all environments
try:
    from legged_gym_isaaclab import tasks  # noqa: F401
except ImportError:
    pass  # Isaac Lab may not be installed yet
