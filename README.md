# Legged Gym → Isaac Lab 迁移项目

本项目将原始的 **legged_gym**--https://github.com/leggedrobotics/legged_gym (基于 Isaac Gym) 完整迁移到 **NVIDIA Isaac Lab** 框架，实现了四足机器人 ANYmal-C 在崎岖地形上的强化学习训练。

## 项目概述

### 实现功能

- **完整的 ANYmal-C 四足机器人仿真环境**
- **基于 PPO 算法的强化学习训练** (使用 RSL-RL 库)
- **多种奖励函数** (速度控制、姿态稳定、能量优化等)
- **GPU 加速训练** (RTX 4060 )

## 迁移过程

### 原始环境

- **原框架**: legged_gym (基于 NVIDIA Isaac Gym Preview)
- **原 API**: VecTask, Gym-style environment
- **原资源格式**: URDF + MJF meshes

### 目标环境

- **新框架**: NVIDIA Isaac Lab (基于 Isaac Sim 4.5)
- **新 API**: DirectRLEnv, USD-based assets

### 核心修改内容

#### 1. 环境基类迁移 (`legged_robot_env.py`)

**修改项**:

- 从 `VecEnvWrapper` 迁移到 `DirectRLEnv`
- 重写初始化流程，使用 Isaac Lab 的场景配置
- 修改观测空间和动作空间定义
- 实现 Isaac Lab 的 reset 机制 (`_reset_idx()`)
- 修改 step 函数返回值 (5-tuple → 4-tuple)
- 添加 `dt` 属性和 `action_scale` 参数
- 修复接触力检测逻辑 (`contact_forces` 初始化)

**关键代码变更**:

```python
# 旧 API (Isaac Gym)
class LeggedRobot(VecTask):
    def reset(self):
        # 使用 Gym API
  
# 新 API (Isaac Lab)  
class LeggedRobotEnv(DirectRLEnv):
    def _reset_idx(self, env_ids):
        # 使用 Isaac Lab 的重置机制
        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids)
```

#### 2. 配置文件重构 (`legged_robot_cfg.py`)

**修改项**:

- 从单一配置类拆分为 `EnvCfg` + `EventCfg` + `RewardsCfg` 等
- 执行器配置: `ImplicitActuatorCfg` → `DCMotorCfg`
- 添加 Isaac Lab 风格的传感器配置 (`ContactSensorCfg`)
- 重写观测空间和动作空间配置
- 调整奖励函数权重初始化顺序

**配置结构变化**:

```python
# 旧结构
class LeggedRobotCfg:
    env: EnvCfg
    terrain: TerrainCfg
    # 所有配置在一个类中

# 新结构
@configclass
class LeggedRobotEnvCfg(DirectRLEnvCfg):
    # 场景配置
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # 观测配置
    observations: ObservationsCfg = ObservationsCfg()
    # 动作配置
    actions: ActionsCfg = ActionsCfg()
    # 奖励配置
    rewards: RewardsCfg = RewardsCfg()
    # 事件配置
    events: EventCfg = EventCfg()
```

#### 3. 训练脚本适配 (`train.py`)

**修改项**:

- 环境包装器适配 RSL-RL 接口
- 修复观测和动作缓冲区访问
- 添加日志目录自动创建
- 适配 GPU/CPU 设备选择

**包装器实现**:

```python
class RslRlVecEnvWrapper(gym.Wrapper):
    """包装 Isaac Lab 环境以兼容 RSL-RL"""
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.num_obs = self.unwrapped.observation_space.shape[1]
        self.num_actions = self.unwrapped.action_space.shape[1]
```

#### 4. assets和机器人配置

**修改项**:

- URDF 资源路径适配
- 关节和执行器映射更新

#### 5. 其他修改

 **API 兼容性修复**:

- `self.num_actions` → `self.unwrapped.num_actions`
- `self.gravity_vec` → `self.scene.env_origins` (重力向量获取)
- `self.commands` → 命令缓冲区初始化
- 奖励缩放因子初始化顺序调整
- 观测缓冲区和动作缓冲区访问路径修正

## 使用环境

### 硬件配置

- **GPU**: NVIDIA GeForce RTX 4060 Laptop 8GB VRAM

### 环境

- **Isaac Sim**: 4.5.0
- **Isaac Lab**: 0.47.1
- **Python**: 3.10
- **关键依赖**:
  - `omni.isaac.lab` - Isaac Lab 核心库
  - `rsl_rl` - RSL 强化学习库
  - `torch` - PyTorch 深度学习框架
  - `tensorboard` - 训练可视化工具


### 训练结果

所有训练结果保存在 `result/` 文件夹中

## 项目结构

```
legged_gym/
├── legged_gym_isaaclab/              # Isaac Lab 扩展 (核心代码)
│   ├── __init__.py                   # 扩展入口
│   ├── extension.toml                # 扩展配置
│   ├── scripts/
│   │   └── train.py                  # 训练脚本 (包含 RSL-RL 包装器)
│   └── tasks/
│       └── locomotion/
│           ├── __init__.py           # 任务注册
│           ├── legged_robot_cfg.py   # 环境配置 (奖励、观测、动作)
│           └── legged_robot_env.py   # 环境实现 (核心逻辑)
│
├── resources/                         # 机器人资产
│   ├── robots/
│   │   └── anymal_c/
│   │       ├── urdf/anymal_c.urdf    # 机器人模型
│   │       └── meshes/               # 3D 网格文件
│   └── actuator_nets/                # 执行器神经网络
│
├── result/                            # 训练结果 (模型检查点和TensorBoard日志)
│
├── train_headless.ps1                 # 无可视化训练脚本
├── train_with_visualization.ps1       # 可视化训练脚本
└── README.md                          # 本文件
```

## 配置说明

### 环境参数 (`legged_robot_cfg.py`)

```python
# 场景配置
scene.num_envs = 4              # 并行环境数量 (影响训练速度)
scene.env_spacing = 2.5         # 环境间距 (米)

# 仿真参数
decimation = 4                  # 控制频率降采样 (仿真步数/控制步数)
episode_length_s = 20           # 回合时长 (秒)

# 动作参数
action_scale = 0.25             # 动作缩放因子
```

### 奖励函数配置

所有奖励函数及其权重在 `RewardsCfg` 中定义:

| 奖励项            | 权重     | 说明                        |
| ----------------- | -------- | --------------------------- |
| `lin_vel_z`     | -2.0     | 惩罚垂直速度 (鼓励水平移动) |
| `ang_vel_xy`    | -0.05    | 惩罚身体旋转                |
| `orientation`   | -0.0     | 惩罚非水平姿态              |
| `base_height`   | -0.0     | 维持目标高度                |
| `torques`       | -0.00001 | 惩罚高扭矩 (能量优化)       |
| `dof_vel`       | -0.0     | 惩罚高关节速度              |
| `dof_acc`       | -2.5e-7  | 惩罚高关节加速度            |
| `action_rate`   | -0.01    | 惩罚动作变化率 (平滑运动)   |
| `collision`     | -1.0     | 惩罚自碰撞                  |
| `termination`   | -0.0     | 惩罚回合终止                |
| `feet_air_time` | 1.0      | 奖励足部腾空时间            |

**自定义奖励**: 修改 `legged_robot_cfg.py` 中的 `rewards.scales` 字典。

### 训练超参数

默认 PPO 参数 (适用于 RSL-RL):

- **学习率**: 1e-3
- **批量大小**: 根据环境数量自动调整
- **迭代次数**: 100 (可通过脚本参数修改)
