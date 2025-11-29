# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for legged robot environment (migrated from legged_gym)."""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetLSTMCfg, DCMotorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


##
# Pre-defined robot configurations
##

# ANYmal-C Actuator (使用与Isaac Lab相同的配置)
ANYDRIVE_3_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,
    stiffness={".*": 80.0},  # legged_gym 使用 80
    damping={".*": 2.0},     # legged_gym 使用 2
)

# ANYmal-C 配置 (匹配 legged_gym 的 URDF)
ANYMAL_C_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # legged_gym 设置为 False
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,
            ".*F_HFE": 0.4,   # Front HFE
            ".*H_HFE": -0.4,  # Hind HFE
            ".*F_KFE": -0.8,  # Front KFE
            ".*H_KFE": 0.8,   # Hind KFE
        },
    ),
    actuators={"legs": ANYDRIVE_3_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)


##
# Environment configuration
##

@configclass
class CommandsCfg:
    """Command specifications matching legged_gym."""
    
    @configclass
    class RangesCfg:
        """Command ranges."""
        lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        lin_vel_y: tuple[float, float] = (-1.0, 1.0)
        ang_vel_yaw: tuple[float, float] = (-1.0, 1.0)
        heading: tuple[float, float] = (-3.14, 3.14)
    
    resampling_time: float = 10.0  # Time before command resampling [s]
    heading_command: bool = False  # If True, compute ang vel from heading error
    ranges: RangesCfg = RangesCfg()


@configclass
class ControlCfg:
    """Control configuration matching legged_gym."""
    control_type: str = "P"  # P: position, V: velocity, T: torques
    stiffness: float = 80.0  # PD stiffness [N*m/rad]
    damping: float = 2.0  # PD damping [N*m*s/rad]
    action_scale: float = 0.5  # Action scaling
    decimation: int = 4  # Decimation of control actions
    

@configclass  
class NormalizationCfg:
    """Normalization configuration."""
    
    @configclass
    class ObsScalesCfg:
        """Observation scaling factors."""
        lin_vel: float = 2.0
        ang_vel: float = 0.25
        dof_pos: float = 1.0
        dof_vel: float = 0.05
        commands: float = 1.0
        
    clip_observations: float = 100.0
    clip_actions: float = 100.0
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    

@configclass
class NoiseCfg:
    """Noise configuration."""
    
    @configclass
    class NoiseScalesCfg:
        """Noise scales for observations."""
        lin_vel: float = 0.1
        ang_vel: float = 0.2
        gravity: float = 0.05
        dof_pos: float = 0.01
        dof_vel: float = 1.5
        
    add_noise: bool = True
    noise_level: float = 1.0  # Scales noise
    noise_scales: NoiseScalesCfg = NoiseScalesCfg()


@configclass
class RewardsCfg:
    """Reward configuration."""
    
    @configclass
    class ScalesCfg:
        """Reward term scales (matching legged_gym)."""
        # Tracking rewards
        tracking_lin_vel: float = 1.0
        tracking_ang_vel: float = 0.5
        
        # Regularization rewards
        lin_vel_z: float = -2.0
        ang_vel_xy: float = -0.05
        orientation: float = -0.0
        torques: float = -0.00001
        dof_vel: float = -0.0
        dof_acc: float = -2.5e-7
        action_rate: float = -0.01
        collision: float = -1.0
        stumble: float = -0.0
        stand_still: float = -0.0
        feet_air_time: float = 1.0
    
    only_positive_rewards: bool = True
    scales: ScalesCfg = ScalesCfg()


@configclass
class LeggedRobotEnvCfg(DirectRLEnvCfg):
    """Configuration for the legged robot environment."""
    
    # Basic settings
    decimation: int = 4
    episode_length_s: float = 20.0
    action_space: int = 12
    observation_space: int = 48
    state_space: int = 0
    num_actions: int = 12
    num_observations: int = 48
    num_states: int = 0
    
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        device="cuda:0",  # 使用GPU加速训练
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    # Robot configuration
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # Contact sensor for feet
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )
    
    # Scene configuration  
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )
    
    # Terrain (flat plane by default)
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    
    # Configuration sub-groups
    commands: CommandsCfg = CommandsCfg()
    control: ControlCfg = ControlCfg()
    normalization: NormalizationCfg = NormalizationCfg()
    noise: NoiseCfg = NoiseCfg()
    rewards: RewardsCfg = RewardsCfg()


@configclass
class AnymalCFlatEnvCfg(LeggedRobotEnvCfg):
    """ANYmal-C Flat terrain configuration (matching legged_gym's anymal_c_flat)."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Override specific settings for flat terrain
        self.scene.num_envs = 4096
        self.episode_length_s = 20.0
        
        # Commands
        self.commands.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.ranges.lin_vel_y = (-1.0, 1.0) 
        self.commands.ranges.ang_vel_yaw = (-1.0, 1.0)
        self.commands.resampling_time = 10.0
        
        # Action scale
        self.action_scale = 0.5  # Scale actions before applying to motors
        
        # Rewards (matching anymal_c_flat)
        self.rewards.scales.tracking_lin_vel = 1.0
        self.rewards.scales.tracking_ang_vel = 0.5
        self.rewards.scales.lin_vel_z = -2.0
        self.rewards.scales.ang_vel_xy = -0.05
        self.rewards.scales.orientation = -0.0
        self.rewards.scales.torques = -0.00001
        self.rewards.scales.dof_vel = -0.0
        self.rewards.scales.dof_acc = -2.5e-7
        self.rewards.scales.action_rate = -0.01
        self.rewards.scales.feet_air_time = 1.0
