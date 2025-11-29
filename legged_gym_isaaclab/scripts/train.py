# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Training script for legged_gym environments in Isaac Lab.

Usage:
    # From legged_gym directory
    python -m legged_gym_isaaclab.scripts.train --task Isaac-Legged-Anymal-C-Flat-v0 --num_envs 4096 --headless
    
    # Or using Isaac Lab's isaaclab.bat
    cd D:\IsaacLab
    .\isaaclab.bat -p d:\legged_gym\legged_gym\legged_gym_isaaclab\scripts\train.py --task Isaac-Legged-Anymal-C-Flat-v0
"""

import argparse
import sys
from datetime import datetime

# Add legged_gym to path
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Isaac Lab imports
try:
    import isaaclab.app
    
    # Create argparser
    parser = argparse.ArgumentParser(description="Train a legged_gym policy with Isaac Lab.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default="Isaac-Legged-Anymal-C-Flat-v0", help="Name of the task.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

    # Append AppLauncher cli args
    isaaclab.app.AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    # Always enable cameras for recording
    if args_cli.video:
        args_cli.enable_cameras = True

    # Launch Isaac Sim
    app_launcher = isaaclab.app.AppLauncher(args_cli)
    simulation_app = app_launcher.app

except ImportError as e:
    print(f"ERROR: Failed to import Isaac Lab: {e}")
    print("Make sure you're running this script with Isaac Lab's Python:")
    print("  D:\\IsaacLab\\isaaclab.bat -p <script_path>")
    sys.exit(1)

# Standard imports after launching Isaac Sim
import gymnasium as gym
import torch

# Isaac Lab imports
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg

# RSL-RL imports
from rsl_rl.runners import OnPolicyRunner

# Import our tasks (this registers them with gymnasium)
import legged_gym_isaaclab.tasks  # noqa: F401


def main():
    """Train with RSL-RL."""
    
    # Parse hydra arguments
    from omegaconf import DictConfig
    import hydra
    from hydra import compose, initialize
    
    # Print info
    print("="*80)
    print(f"Training legged_gym task: {args_cli.task}")
    print("="*80)
    
    # Create environment
    env_cfg_class = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
    env_cfg = env_cfg_class()  # 实例化配置
    
    # Override num_envs if specified
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
        
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        
    # Print environment info
    print(f"[INFO] Created environment: {args_cli.task}")
    print(f"[INFO] Number of environments: {env.unwrapped.num_envs}")
    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")
    
    # Create RSL-RL configuration
    from rsl_rl.runners import OnPolicyRunner
    
    # Default RSL-RL config for legged robots
    agent_cfg = {
        "class_name": "ActorCritic",
        "init_noise_std": 1.0,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
    }
    
    algorithm_cfg = {
        "class_name": "PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.01,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    }
    
    runner_cfg = {
        "seed": args_cli.seed if args_cli.seed is not None else 42,
        "device": env.unwrapped.device,
        "num_steps_per_env": 24,
        "max_iterations": args_cli.max_iterations if args_cli.max_iterations is not None else 1500,
        "empirical_normalization": False,
        "policy": agent_cfg,
        "algorithm": algorithm_cfg,
        "experiment_name": "legged_gym_anymal_c_flat",
        "run_name": f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "logger": "tensorboard",
        "neptune_project": "isaaclab",
        "wandb_project": "isaaclab",
        "resume": False,
        "load_run": None,
        "load_checkpoint": None,
        "save_interval": 50,  # Save checkpoint every 50 iterations
        "log_interval": 1,    # Log every iteration
        "obs_groups": {},  # RSL-RL requires this parameter (can be empty dict)
    }
    
    # Create runner
    from omegaconf import OmegaConf
    runner_cfg_obj = OmegaConf.create(runner_cfg)
    
    print("[INFO] Creating RSL-RL runner...")
    # Unwrap the environment to get the actual DirectRLEnv instance
    # RSL-RL expects env.get_observations() method which Gymnasium wrappers don't have
    unwrapped_env = env.unwrapped
    
    # Create a wrapper that converts DirectRLEnv's 5-tuple return to RSL-RL's expected 4-tuple
    class RSLRLCompatEnv:
        """Wrapper to make DirectRLEnv compatible with RSL-RL's interface"""
        def __init__(self, env):
            self.env = env
            self.device = env.device
            
        def step(self, actions):
            # DirectRLEnv returns (obs, reward, terminated, truncated, extras)
            # RSL-RL expects (obs, reward, dones, extras)
            obs, reward, terminated, truncated, extras = self.env.step(actions)
            # Combine terminated and truncated into dones
            dones = terminated | truncated
            return obs, reward, dones, extras
            
        def reset(self, *args, **kwargs):
            return self.env.reset(*args, **kwargs)
            
        def get_observations(self):
            return self.env.get_observations()
            
        def __getattr__(self, name):
            return getattr(self.env, name)
    
    wrapped_env = RSLRLCompatEnv(unwrapped_env)
    
    # Create log directory
    import os
    log_dir = os.path.join("logs", "legged_gym_isaaclab", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")
    
    runner = OnPolicyRunner(wrapped_env, runner_cfg_obj, log_dir=log_dir, device=runner_cfg["device"])
    
    # Start training
    print("="*80)
    print("Starting training...")
    print("="*80)
    runner.learn(num_learning_iterations=runner_cfg["max_iterations"], init_at_random_ep_len=True)
    
    # Close environment
    env.close()
    
    print("="*80)
    print("Training completed!")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Print exception and traceback
        import traceback
        print("="*80)
        print("ERROR during training:")
        print("="*80)
        traceback.print_exc()
        raise
    finally:
        # Close simulation
        simulation_app.close()
