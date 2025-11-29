# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Legged Robot environment for Isaac Lab (migrated from legged_gym).

This module provides the base class for legged robot locomotion tasks,
replicating the behavior of the original legged_gym implementation.
"""

import torch
from typing import Dict

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_rotate_inverse, wrap_to_pi

from .legged_robot_cfg import LeggedRobotEnvCfg


class ObservationDict(dict):
    """
    A dictionary subclass that supports .to() method for RSL-RL compatibility.
    RSL-RL calls obs.to(device) in the training loop.
    """
    def to(self, device):
        """Move all tensor values to the specified device."""
        return ObservationDict({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in self.items()})


class LeggedRobotEnv(DirectRLEnv):
    """
    Legged robot locomotion environment adapted from legged_gym.
    
    Migrates the core functionality of legged_gym's LeggedRobot class to Isaac Lab's DirectRLEnv.
    """
    
    cfg: LeggedRobotEnvCfg
    
    def __init__(self, cfg: LeggedRobotEnvCfg, render_mode: str | None = None, **kwargs):
        # Setup reward scales BEFORE calling super().__init__()
        # because _init_custom_buffers() is called during super().__init__()
        self.reward_scales = {}
        for key in dir(cfg.rewards.scales):
            if not key.startswith("_"):
                self.reward_scales[key] = getattr(cfg.rewards.scales, key)
        
        # Command ranges
        self.command_ranges = {
            "lin_vel_x": cfg.commands.ranges.lin_vel_x,
            "lin_vel_y": cfg.commands.ranges.lin_vel_y,
            "ang_vel_yaw": cfg.commands.ranges.ang_vel_yaw,
            "heading": cfg.commands.ranges.heading if hasattr(cfg.commands.ranges, "heading") else [-3.14, 3.14],
        }
        
        # PD gains (硬编码为 legged_gym 的默认值)
        self.p_gains_val = 80.0
        self.d_gains_val = 2.0
        
        # Gravity vector will be initialized after super().__init__() when device is available
        self._gravity_vec_init = False
        
        # Call parent class initializer (this will call _setup_scene and _init_custom_buffers)
        super().__init__(cfg, render_mode, **kwargs)
        
        # Simulation timestep (physics dt * decimation)
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        
        # Initialize PD gain tensors after device is set
        # RSL-RL expects num_actions attribute
        self.num_actions = self.cfg.action_space if isinstance(self.cfg.action_space, int) else 12
        self.p_gains = torch.ones(self.num_actions, device=self.device) * self.p_gains_val
        self.d_gains = torch.ones(self.num_actions, device=self.device) * self.d_gains_val
        
        # Initialize gravity vector
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat((self.num_envs, 1))
        
        # Initialize custom buffers (commands, last_actions, etc.)
        self._init_custom_buffers()
        
        print(f"[LeggedRobotEnv] Initialized with {self.num_envs} environments")
        print(f"[LeggedRobotEnv] Episode length: {self.max_episode_length} steps ({self.max_episode_length_s}s)")
        
    def _setup_scene(self):
        """Setup the scene entities (robot, terrain, etc.)"""
        # Create robot articulation
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        
        # Create and add contact sensor
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        
        # Setup terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step"""
        self.actions = actions.clone()
        
        # Compute torques using PD controller (replicating legged_gym)
        torques = self._compute_torques(actions)
        
        # Apply torques to robot
        self.robot.set_joint_effort_target(torques)
        
    def _apply_action(self):
        """Already handled in _pre_physics_step"""
        pass
    
    def get_observations(self) -> tuple:
        """
        Get current observations for RSL-RL compatibility.
        Returns tuple of (observations, extras_dict).
        """
        obs = self._get_observations()
        extras = {}
        if hasattr(self, 'extras'):
            extras = self.extras
        return obs, extras
        
    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute torques from actions using PD controller.
        Migrated from legged_gym's _compute_torques method.
        """
        actions_scaled = actions * self.cfg.action_scale
        control_type = getattr(self.cfg.control, 'control_type', 'P')
        
        if control_type == "P":
            # Position control
            torques = (
                self.p_gains * (actions_scaled + self.default_dof_pos - self.robot.data.joint_pos)
                - self.d_gains * self.robot.data.joint_vel
            )
        elif control_type == "V":
            # Velocity control
            torques = (
                self.p_gains * (actions_scaled - self.robot.data.joint_vel)
                - self.d_gains * (self.robot.data.joint_vel - self.last_dof_vel) / self.dt
            )
        elif control_type == "T":
            # Direct torque control
            torques = actions_scaled
        else:
            raise ValueError(f"Unknown control type: {control_type}")
            
        # Clip to torque limits
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
        
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """
        Compute observations.
        Replicates legged_gym's compute_observations method.
        """
        # Get robot body velocities in body frame
        base_lin_vel = quat_rotate_inverse(self.robot.data.root_quat_w, self.robot.data.root_lin_vel_w)
        base_ang_vel = quat_rotate_inverse(self.robot.data.root_quat_w, self.robot.data.root_ang_vel_w)
        
        # Projected gravity
        projected_gravity = quat_rotate_inverse(self.robot.data.root_quat_w, self.gravity_vec)
        
        # Build observation (matching legged_gym structure)
        obs = torch.cat([
            base_lin_vel * self.cfg.normalization.obs_scales.lin_vel,
            base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,
            projected_gravity,
            self.commands[:, :3] * self.cfg.normalization.obs_scales.commands,
            (self.robot.data.joint_pos - self.default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos,
            self.robot.data.joint_vel * self.cfg.normalization.obs_scales.dof_vel,
            self.actions,
        ], dim=-1)
        
        # Add noise if enabled
        if self.cfg.noise.add_noise:
            obs += (2 * torch.rand_like(obs) - 1) * self.noise_scale_vec
            
        # Clip observations
        obs = torch.clip(obs, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)
        
        # Cache to obs_buf for RSL-RL compatibility
        self.obs_buf = obs
        
        return ObservationDict({"policy": obs})
    
    def get_observations(self):
        """
        RSL-RL compatibility method.
        Returns the cached observation dict with 'policy' key.
        RSL-RL's runner will extract the tensor value using obs["policy"].to(device)
        """
        return ObservationDict({"policy": self.obs_buf})
        
    def _get_rewards(self) -> torch.Tensor:
        """
        Compute rewards.
        Migrated from legged_gym's compute_reward method.
        """
        total_reward = torch.zeros(self.num_envs, device=self.device)
        
        # Track all reward components (for logging)
        for name, scale in self.reward_scales.items():
            if scale == 0:
                continue
            reward_func = getattr(self, f"_reward_{name}", None)
            if reward_func is not None:
                rew = reward_func() * scale
                total_reward += rew
                self.episode_sums[name] += rew
                
        # Clip to positive if configured
        if self.cfg.rewards.only_positive_rewards:
            total_reward = torch.clip(total_reward, min=0.0)
            
        return total_reward
        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions"""
        # Base contact termination using contact sensor
        # Check if base is in contact with ground
        base_contact = torch.any(
            self.contact_sensor.data.net_forces_w_history[:, 0, self.termination_contact_indices, 2] > 1.0,
            dim=1
        )
        
        # Time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Combine terminations
        terminated = base_contact
        truncated = time_out
        
        return terminated, truncated
        
    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments"""
        super()._reset_idx(env_ids)
        
        # Reset robot joints (matching legged_gym's _reset_dofs)
        joint_pos = self.default_dof_pos[env_ids] * torch.rand(
            len(env_ids), self.robot.num_joints, device=self.device
        ) * 0.5 + 0.75  # Random between 0.75 and 1.25 times default
        joint_vel = torch.zeros(len(env_ids), self.robot.num_joints, device=self.device)
        
        # Write to robot data  
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # Reset robot base (matching legged_gym's _reset_root_states)
        root_pos = self.default_root_pos[env_ids].clone()
        root_pos[:, :2] += torch.rand(len(env_ids), 2, device=self.device) * 2.0 - 1.0  # ±1m in xy
        root_quat = self.default_root_quat[env_ids]
        root_lin_vel = (torch.rand(len(env_ids), 3, device=self.device) - 0.5)  # Random linear velocities
        root_ang_vel = (torch.rand(len(env_ids), 3, device=self.device) - 0.5)  # Random angular velocities
        
        # Write to robot data
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1), env_ids=env_ids
        )
        
        # Resample commands
        self._resample_commands(env_ids)
        
        # Reset episode tracking
        self.last_dof_vel[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        
        # Reset episode sums
        for key in self.episode_sums.keys():
            self.episode_sums[key][env_ids] = 0.0
            
    def _resample_commands(self, env_ids: torch.Tensor):
        """Resample commands for given environments"""
        self.commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * \
            (self.command_ranges["lin_vel_x"][1] - self.command_ranges["lin_vel_x"][0]) + \
            self.command_ranges["lin_vel_x"][0]
        self.commands[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * \
            (self.command_ranges["lin_vel_y"][1] - self.command_ranges["lin_vel_y"][0]) + \
            self.command_ranges["lin_vel_y"][0]
        self.commands[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * \
            (self.command_ranges["ang_vel_yaw"][1] - self.command_ranges["ang_vel_yaw"][0]) + \
            self.command_ranges["ang_vel_yaw"][0]
            
        # Set small commands to zero
        lin_vel_norm = torch.norm(self.commands[env_ids, :2], dim=1)
        self.commands[env_ids, :2] *= (lin_vel_norm > 0.2).unsqueeze(1)
        
    def _init_custom_buffers(self):
        """Initialize custom buffers matching legged_gym"""
        # Observation buffer for RSL-RL compatibility
        self.obs_buf = torch.zeros(self.num_envs, self.cfg.observation_space, device=self.device)
        
        # Commands [lin_vel_x, lin_vel_y, ang_vel_yaw]
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Tracking buffers
        self.last_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self.last_dof_vel = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        self.feet_air_time = torch.zeros(self.num_envs, 4, device=self.device)  # Assuming 4 feet
        
        # Episode sums for logging
        self.episode_sums = {}
        for key in self.reward_scales.keys():
            self.episode_sums[key] = torch.zeros(self.num_envs, device=self.device)
            
        # Default positions
        self.default_dof_pos = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        self.default_root_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.default_root_pos[:, 2] = 0.6  # Default height
        self.default_root_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.default_root_quat[:, 0] = 1.0  # w=1 for identity quaternion
        
        # Torque limits (will be set from config)
        self.torque_limits = torch.ones(self.robot.num_joints, device=self.device) * 80.0  # Default 80 Nm
        
        # Gravity vector
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        
        # Noise scale vector
        self.noise_scale_vec = self._get_noise_scale_vec()
        
        # Termination contact body indices (will be set after robot is created)
        self.termination_contact_indices = torch.tensor([0], device=self.device)  # Base link
        
    def _get_noise_scale_vec(self) -> torch.Tensor:
        """Create noise scale vector for observations"""
        noise_vec = torch.zeros(self.cfg.num_observations, device=self.device)
        
        if not self.cfg.noise.add_noise:
            return noise_vec
            
        scales = self.cfg.noise.noise_scales
        level = self.cfg.noise.noise_level
        obs_scales = self.cfg.normalization.obs_scales
        
        # Match legged_gym observation structure
        idx = 0
        noise_vec[idx:idx+3] = scales.lin_vel * level * obs_scales.lin_vel; idx += 3
        noise_vec[idx:idx+3] = scales.ang_vel * level * obs_scales.ang_vel; idx += 3
        noise_vec[idx:idx+3] = scales.gravity * level; idx += 3
        noise_vec[idx:idx+3] = 0.0; idx += 3  # Commands (no noise)
        noise_vec[idx:idx+12] = scales.dof_pos * level * obs_scales.dof_pos; idx += 12
        noise_vec[idx:idx+12] = scales.dof_vel * level * obs_scales.dof_vel; idx += 12
        noise_vec[idx:idx+12] = 0.0  # Actions (no noise)
        
        return noise_vec
        
    # ==================== Reward Functions ====================
    # Migrated from legged_gym's reward methods
    
    def _reward_tracking_lin_vel(self) -> torch.Tensor:
        """Reward for tracking linear velocity commands"""
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - quat_rotate_inverse(self.robot.data.root_quat_w, self.robot.data.root_lin_vel_w)[:, :2]
        ), dim=1)
        return torch.exp(-lin_vel_error / 0.25)
        
    def _reward_tracking_ang_vel(self) -> torch.Tensor:
        """Reward for tracking angular velocity commands"""
        ang_vel_error = torch.square(
            self.commands[:, 2] - quat_rotate_inverse(self.robot.data.root_quat_w, self.robot.data.root_ang_vel_w)[:, 2]
        )
        return torch.exp(-ang_vel_error / 0.25)
        
    def _reward_lin_vel_z(self) -> torch.Tensor:
        """Penalize z-axis linear velocity"""
        return torch.square(
            quat_rotate_inverse(self.robot.data.root_quat_w, self.robot.data.root_lin_vel_w)[:, 2]
        )
        
    def _reward_ang_vel_xy(self) -> torch.Tensor:
        """Penalize xy-axis angular velocity"""
        return torch.sum(torch.square(
            quat_rotate_inverse(self.robot.data.root_quat_w, self.robot.data.root_ang_vel_w)[:, :2]
        ), dim=1)
        
    def _reward_orientation(self) -> torch.Tensor:
        """Penalize non-flat base orientation"""
        projected_gravity = quat_rotate_inverse(self.robot.data.root_quat_w, self.gravity_vec)
        return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
        
    def _reward_torques(self) -> torch.Tensor:
        """Penalize torque magnitude"""
        return torch.sum(torch.square(self.robot.data.applied_torque), dim=1)
        
    def _reward_dof_vel(self) -> torch.Tensor:
        """Penalize joint velocities"""
        return torch.sum(torch.square(self.robot.data.joint_vel), dim=1)
        
    def _reward_dof_acc(self) -> torch.Tensor:
        """Penalize joint accelerations"""
        return torch.sum(torch.square(
            (self.robot.data.joint_vel - self.last_dof_vel) / self.dt
        ), dim=1)
        
    def _reward_action_rate(self) -> torch.Tensor:
        """Penalize action rate of change"""
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        
    def _reward_collision(self) -> torch.Tensor:
        """Penalize collisions on specific bodies"""
        # This needs body-specific contact forces (to be implemented)
        return torch.zeros(self.num_envs, device=self.device)
        
    def _reward_stumble(self) -> torch.Tensor:
        """Penalize foot stumbling"""
        # This needs foot contact detection (to be implemented)
        return torch.zeros(self.num_envs, device=self.device)
        
    def _reward_stand_still(self) -> torch.Tensor:
        """Reward for standing still when commanded"""
        # Check if all command velocities are near zero
        return torch.sum(torch.abs(self.commands[:, :3]) < 0.1, dim=1) == 3
        
    def _reward_feet_air_time(self) -> torch.Tensor:
        """Reward for feet air time"""
        # This needs contact sensing (to be implemented)
        return torch.zeros(self.num_envs, device=self.device)
