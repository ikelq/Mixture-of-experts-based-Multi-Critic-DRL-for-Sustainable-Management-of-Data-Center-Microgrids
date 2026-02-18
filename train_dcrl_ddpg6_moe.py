#!/usr/bin/env python3
"""
DDPG-MOE训练DCRL环境的脚本
基于cleanRL的DDPG实现，适配DCRL环境
改进版本：使用共享表示的三critic网络架构（MOE Multi-Critic）
- 共享编码器：提取通用特征表示
- 三个任务特定头部：Energy, Carbon, Penalty
- 确定性策略网络
- 用于验证MOE-MC框架对DDPG的通用性 (R1-2)
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# 导入DCRL环境
from dcrl_env_lab2 import DCRL


@dataclass
class Args:
    exp_name: str = "dcrl_ddpg6_moe"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "dcrl-ddpg-moe"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""

    # Algorithm specific arguments
    total_timesteps: int = int(1e6)
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 2400
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    target_network_frequency: int = 1
    """the frequency of updates for the target networks"""
    noise_scale: float = 0.1
    """exploration noise scale"""

    # Shared Critic specific arguments
    energy_weight: float = 1.0
    """weight for energy cost critic"""
    carbon_weight: float = 1.0
    """weight for carbon footprint critic"""
    penalty_weight: float = 1.0
    """weight for penalty critic"""

    # Evaluation arguments
    eval_frequency: int = 2400
    """frequency of evaluation episodes"""
    eval_episodes: list = range(0, 360, 10)
    """evaluation episode day IDs"""

    # MoE specific arguments
    num_experts: int = 5
    """number of experts in the MoE architecture"""


class MultiRewardReplayBuffer:
    """自定义ReplayBuffer，用于存储多个sub-rewards"""
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: torch.device,
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.n_envs = n_envs
        
        obs_shape = observation_space.shape
        action_shape = action_space.shape
        
        self.observations = np.zeros((buffer_size, n_envs) + obs_shape, dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, n_envs) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_envs) + action_shape, dtype=np.float32)
        
        # 存储三个sub-rewards
        self.energy_rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.carbon_rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.penalty_rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.total_rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        
        self.pos = 0
        self.full = False
        
    def add(self, obs, next_obs, actions, rewards, dones, infos):
        """添加数据到buffer，从infos中提取sub-rewards"""
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(actions).copy()
        self.dones[self.pos] = np.array(dones).copy()
        
        # 从infos中提取sub-rewards
        energy_rewards = np.zeros(self.n_envs)
        carbon_rewards = np.zeros(self.n_envs)
        penalty_rewards = np.zeros(self.n_envs)
        
        for i in range(self.n_envs):
            if isinstance(infos, dict):
                info = infos
                idx = 0 if self.n_envs == 1 else i
            else:
                info = infos[i] if i < len(infos) and infos[i] is not None else {}
                idx = i
                
            if 'final_info' in info:
                final_info = info['final_info'][0]
                energy_rewards[idx] = final_info['energy_reward']
                carbon_rewards[idx] = final_info['carbon_reward']
                penalty_rewards[idx] = final_info['workload_reward']
            else:
                energy_rewards[idx] = info.get('energy_reward', 0)
                carbon_rewards[idx] = info.get('carbon_reward', 0)
                penalty_rewards[idx] = info.get('workload_reward', 0)
        
        self.energy_rewards[self.pos] = energy_rewards
        self.carbon_rewards[self.pos] = carbon_rewards
        self.penalty_rewards[self.pos] = penalty_rewards
        
        total_rewards = energy_rewards + carbon_rewards + penalty_rewards
        self.total_rewards[self.pos] = total_rewards
        
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True
    
    def sample(self, batch_size):
        """采样batch数据"""
        if self.full:
            indices = np.random.randint(0, self.buffer_size, size=batch_size)
        else:
            indices = np.random.randint(0, self.pos, size=batch_size)
        
        env_indices = np.random.randint(0, self.n_envs, size=batch_size)
        
        batch_obs = torch.FloatTensor(self.observations[indices, env_indices]).to(self.device)
        batch_next_obs = torch.FloatTensor(self.next_observations[indices, env_indices]).to(self.device)
        batch_actions = torch.FloatTensor(self.actions[indices, env_indices]).to(self.device)
        batch_rewards = torch.FloatTensor(self.total_rewards[indices, env_indices]).to(self.device)
        batch_dones = torch.FloatTensor(self.dones[indices, env_indices]).to(self.device)
        
        batch_energy_rewards = torch.FloatTensor(self.energy_rewards[indices, env_indices]).to(self.device)
        batch_carbon_rewards = torch.FloatTensor(self.carbon_rewards[indices, env_indices]).to(self.device)
        batch_penalty_rewards = torch.FloatTensor(self.penalty_rewards[indices, env_indices]).to(self.device)
        
        class ReplayBufferSamples:
            def __init__(self, observations, next_observations, actions, rewards, dones, 
                        energy_rewards, carbon_rewards, penalty_rewards):
                self.observations = observations
                self.next_observations = next_observations
                self.actions = actions
                self.rewards = rewards
                self.dones = dones
                self.energy_rewards = energy_rewards
                self.carbon_rewards = carbon_rewards
                self.penalty_rewards = penalty_rewards
        
        return ReplayBufferSamples(
            batch_obs, batch_next_obs, batch_actions, batch_rewards, batch_dones,
            batch_energy_rewards, batch_carbon_rewards, batch_penalty_rewards
        )


class DCRLWrapper(gym.Env):
    """Wrapper to make DCRL environment compatible with gym.vector interface"""
    def __init__(self):
        self.env = DCRL()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.episode_return = 0
        self.episode_length = 0
        
    def reset(self, day_id=None, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            self.env.workload_env.action_space.seed(seed)
            self.env.dc_env.action_space.seed(seed)
            self.env.battery_env.action_space.seed(seed)
            
        self.episode_return = 0
        self.episode_length = 0
        return self.env.reset(day_id=day_id)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_return += reward
        self.episode_length += 1
        
        if terminated or truncated:
            info['episode'] = {
                'r': self.episode_return,
                'l': self.episode_length
            }
            
        return obs, reward, terminated, truncated, info


class ExpertNetwork(nn.Module):
    """专家网络 - 每个专家专注于特定的任务模式"""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class GatingNetwork(nn.Module):
    """门控网络 - 动态选择或组合专家输出"""
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )

    def forward(self, x):
        return self.net(x)


class SharedEncoder(nn.Module):
    """共享编码器网络，使用MoE结构 - 每个任务有独立的门控"""
    def __init__(self, obs_dim, action_dim, num_experts=5):
        super().__init__()
        self.num_experts = num_experts
        input_dim = obs_dim + action_dim
        
        # 创建多个专家网络
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim) for _ in range(num_experts)
        ])
        
        # 为每个任务创建独立的门控网络
        self.energy_gating = GatingNetwork(input_dim, num_experts)
        self.carbon_gating = GatingNetwork(input_dim, num_experts)
        self.penalty_gating = GatingNetwork(input_dim, num_experts)
        
        # 动态负载均衡参数 - 每个任务独立
        self.register_buffer('energy_expert_usage', torch.zeros(num_experts))
        self.register_buffer('carbon_expert_usage', torch.zeros(num_experts))
        self.register_buffer('penalty_expert_usage', torch.zeros(num_experts))
        # 固定温度参数（每个head一个）
        self.register_buffer('energy_temperature', torch.ones(1) * 1.0)
        self.register_buffer('carbon_temperature', torch.ones(1) * 1.0)
        self.register_buffer('penalty_temperature', torch.ones(1) * 1.0)
        self.usage_decay = 0.99

    def update_usage(self, gate_weights_dict):
        """更新所有任务的专家使用率统计"""
        with torch.no_grad():
            for task_type, gate_weights in gate_weights_dict.items():
                current_usage = gate_weights.mean(dim=0)
                if task_type == 'energy':
                    self.energy_expert_usage = self.usage_decay * self.energy_expert_usage + (1 - self.usage_decay) * current_usage
                elif task_type == 'carbon':
                    self.carbon_expert_usage = self.usage_decay * self.carbon_expert_usage + (1 - self.usage_decay) * current_usage
                elif task_type == 'penalty':
                    self.penalty_expert_usage = self.usage_decay * self.penalty_expert_usage + (1 - self.usage_decay) * current_usage

    def forward(self, x, a, training=True):
        """一次性计算所有任务的输出"""
        x_a = torch.cat([x, a], 1)
        
        # 获取每个专家的输出（只计算一次）
        expert_outputs = torch.stack([expert(x_a) for expert in self.experts], dim=1)
        
        # 为每个任务计算门控权重
        energy_gate_logits = self.energy_gating(x_a)
        carbon_gate_logits = self.carbon_gating(x_a)
        penalty_gate_logits = self.penalty_gating(x_a)
        
        # 应用各自的温度并获取门控权重
        energy_gate_weights = F.softmax(energy_gate_logits / self.energy_temperature, dim=-1)
        carbon_gate_weights = F.softmax(carbon_gate_logits / self.carbon_temperature, dim=-1)
        penalty_gate_weights = F.softmax(penalty_gate_logits / self.penalty_temperature, dim=-1)
        
        # 只在训练时更新使用率统计
        if training:
            gate_weights_dict = {
                'energy': energy_gate_weights,
                'carbon': carbon_gate_weights,
                'penalty': penalty_gate_weights
            }
            self.update_usage(gate_weights_dict)
        
        # 为每个任务组合专家输出
        energy_features = torch.sum(expert_outputs * energy_gate_weights.unsqueeze(-1), dim=1)
        carbon_features = torch.sum(expert_outputs * carbon_gate_weights.unsqueeze(-1), dim=1)
        penalty_features = torch.sum(expert_outputs * penalty_gate_weights.unsqueeze(-1), dim=1)
        
        return energy_features, carbon_features, penalty_features, energy_gate_weights, carbon_gate_weights, penalty_gate_weights


class TaskHead(nn.Module):
    """任务特定的头部网络"""
    def __init__(self, input_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.head(x)


class QNetwork(nn.Module):
    """Q网络 - 使用MoE结构的共享编码器和任务特定头部"""
    def __init__(self, env, num_experts=5):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)

        # 共享编码器（使用MoE结构）
        self.encoder = SharedEncoder(obs_dim, action_dim, num_experts=num_experts)
        
        # 任务特定头部
        self.energy_head = TaskHead(128)
        self.carbon_head = TaskHead(128)
        self.penalty_head = TaskHead(128)

    def forward(self, x, a, training=True):
        energy_features, carbon_features, penalty_features, energy_gate_weights, carbon_gate_weights, penalty_gate_weights = self.encoder(x, a, training=training)
        
        energy_q = self.energy_head(energy_features)
        carbon_q = self.carbon_head(carbon_features)
        penalty_q = self.penalty_head(penalty_features)
        
        return energy_q, carbon_q, penalty_q, energy_gate_weights, carbon_gate_weights, penalty_gate_weights


class DDPGActor(nn.Module):
    """DDPG的确定性策略网络"""
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.net(x)
    
    def get_action(self, x, noise_scale=0.1, deterministic=False):
        action = self(x)
        if not deterministic:
            noise = torch.randn_like(action) * noise_scale
            action = action + noise
            action = torch.clamp(action, -1, 1)
        return action


if __name__ == "__main__":
    
    args = tyro.cli(Args)
    run_name = f"dcrl__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"PATH: /root/MOE-DRL-DC2/dc-rl_clearnrl3_long2")
    print(f"Using device: {device}")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [lambda: DCRLWrapper() for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Evaluation environment setup
    eval_env = DCRLWrapper()

    print(f"Observation space: {envs.single_observation_space}")
    print(f"Action space: {envs.single_action_space}")
    print(f"Number of experts: {args.num_experts}")

    # 创建网络
    actor = DDPGActor(envs).to(device)
    actor_target = DDPGActor(envs).to(device)
    actor_target.load_state_dict(actor.state_dict())

    # 创建双MOE Q网络 (减少过估计偏差)
    qf1 = QNetwork(envs, num_experts=args.num_experts).to(device)
    qf2 = QNetwork(envs, num_experts=args.num_experts).to(device)
    qf1_target = QNetwork(envs, num_experts=args.num_experts).to(device)
    qf2_target = QNetwork(envs, num_experts=args.num_experts).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    # 优化器
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)

    # 使用自定义的MultiRewardReplayBuffer
    envs.single_observation_space.dtype = np.float32
    rb = MultiRewardReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
    )
    start_time = time.time()

    # Start the game
    obs, _ = envs.reset(seed=args.seed)
    print("Starting training...")
    
    best_episode_return = float('-inf')
    best_model_path = None

    for global_step in range(args.total_timesteps):
        # Action selection
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor.get_action(torch.Tensor(obs).to(device), noise_scale=args.noise_scale)
                actions = actions.cpu().numpy()

        # Execute action
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Evaluation logic
        if global_step % args.eval_frequency == 0:
            eval_returns = []
            eval_lengths = []
            eval_energy_rewards = []
            eval_carbon_rewards = []
            eval_penalty_rewards = []
            
            for day_id in args.eval_episodes:
                eval_obs, _ = eval_env.reset(day_id=day_id, seed=args.seed)
                episode_return = 0
                episode_length = 0
                episode_energy = 0
                episode_carbon = 0
                episode_penalty = 0
                
                while True:
                    with torch.no_grad():
                        action = actor.get_action(torch.Tensor(eval_obs).unsqueeze(0).to(device), deterministic=True)
                        action = action.squeeze(0).cpu().numpy()
                    
                    eval_obs, reward, terminated, truncated, info = eval_env.step(action)
                    episode_return += reward
                    episode_length += 1
                    
                    if 'energy_reward' in info:
                        episode_energy += info['energy_reward']
                        episode_carbon += info['carbon_reward']
                        episode_penalty += info['workload_reward']
                    
                    if terminated or truncated:
                        break
                
                eval_returns.append(episode_return)
                eval_lengths.append(episode_length)
                eval_energy_rewards.append(episode_energy)
                eval_carbon_rewards.append(episode_carbon)
                eval_penalty_rewards.append(episode_penalty)
            
            avg_eval_return = np.mean(eval_returns)
            avg_eval_length = np.mean(eval_lengths)
            avg_eval_energy = np.mean(eval_energy_rewards)
            avg_eval_carbon = np.mean(eval_carbon_rewards)
            avg_eval_penalty = np.mean(eval_penalty_rewards)
            
            print(f"Step {global_step:6d} | Eval Return: {avg_eval_return:8.3f} | Eval Length: {int(avg_eval_length):3d}")
            
            if avg_eval_return > best_episode_return:
                best_episode_return = avg_eval_return
                print(f"New best evaluation return: {best_episode_return:.3f}")
                best_model_path = f"models/{run_name}/best_model"
                os.makedirs(best_model_path, exist_ok=True)
                torch.save(actor.state_dict(), f"{best_model_path}/actor.pth")
                torch.save(qf1.state_dict(), f"{best_model_path}/qf1.pth")
                torch.save(qf2.state_dict(), f"{best_model_path}/qf2.pth")
                print(f"New best model saved to {best_model_path}")
            
            writer.add_scalar("charts/eval_episodic_return", avg_eval_return, global_step)
            writer.add_scalar("charts/eval_episodic_length", avg_eval_length, global_step)
            writer.add_scalar("eval_sub_rewards/energy_avg", avg_eval_energy, global_step)
            writer.add_scalar("eval_sub_rewards/carbon_avg", avg_eval_carbon, global_step)
            writer.add_scalar("eval_sub_rewards/penalty_avg", avg_eval_penalty, global_step)

        # Save data to replay buffer
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # Training
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            with torch.no_grad():
                # 使用目标actor获取下一个动作
                next_actions = actor_target(data.next_observations)

                # 获取双Q网络的目标Q值
                qf1_next_energy, qf1_next_carbon, qf1_next_penalty, _, _, _ = qf1_target(data.next_observations, next_actions)
                qf2_next_energy, qf2_next_carbon, qf2_next_penalty, _, _, _ = qf2_target(data.next_observations, next_actions)

                # 计算每个任务的最小Q值 (减少过估计偏差)
                min_qf_next_energy = torch.min(qf1_next_energy, qf2_next_energy)
                min_qf_next_carbon = torch.min(qf1_next_carbon, qf2_next_carbon)
                min_qf_next_penalty = torch.min(qf1_next_penalty, qf2_next_penalty)

                # 计算目标Q值
                next_energy_q_value = data.energy_rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next_energy.view(-1)
                next_carbon_q_value = data.carbon_rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next_carbon.view(-1)
                next_penalty_q_value = data.penalty_rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next_penalty.view(-1)

                # Composite target Q value
                min_qf_next_composite = torch.min(
                    qf1_next_energy + qf1_next_carbon + qf1_next_penalty,
                    qf2_next_energy + qf2_next_carbon + qf2_next_penalty
                )
                next_q_value = (data.energy_rewards.flatten() + data.carbon_rewards.flatten() + data.penalty_rewards.flatten() +
                               (1 - data.dones.flatten()) * args.gamma * min_qf_next_composite.view(-1))

            # 获取当前Q值和门控权重
            qf1_energy, qf1_carbon, qf1_penalty, qf1_energy_gate_weights, qf1_carbon_gate_weights, qf1_penalty_gate_weights = qf1(data.observations, data.actions)
            qf2_energy, qf2_carbon, qf2_penalty, qf2_energy_gate_weights, qf2_carbon_gate_weights, qf2_penalty_gate_weights = qf2(data.observations, data.actions)

            # 计算每个任务的损失
            qf1_energy_loss = F.mse_loss(qf1_energy.view(-1), next_energy_q_value)
            qf2_energy_loss = F.mse_loss(qf2_energy.view(-1), next_energy_q_value)
            qf1_carbon_loss = F.mse_loss(qf1_carbon.view(-1), next_carbon_q_value)
            qf2_carbon_loss = F.mse_loss(qf2_carbon.view(-1), next_carbon_q_value)
            qf1_penalty_loss = F.mse_loss(qf1_penalty.view(-1), next_penalty_q_value)
            qf2_penalty_loss = F.mse_loss(qf2_penalty.view(-1), next_penalty_q_value)

            # Composite TD loss
            qf1_composite_loss = F.mse_loss(qf1_energy.view(-1) + qf1_carbon.view(-1) + qf1_penalty.view(-1), next_q_value)
            qf2_composite_loss = F.mse_loss(qf2_energy.view(-1) + qf2_carbon.view(-1) + qf2_penalty.view(-1), next_q_value)

            # 计算负载均衡损失
            usage_balance_loss = 0.0
            for gate_weights in [qf1_energy_gate_weights, qf1_carbon_gate_weights, qf1_penalty_gate_weights,
                                qf2_energy_gate_weights, qf2_carbon_gate_weights, qf2_penalty_gate_weights]:
                current_usage = gate_weights.mean(dim=0)
                usage_deviation = torch.abs(current_usage - 1.0/qf1.encoder.num_experts)
                usage_balance_loss += torch.mean(usage_deviation)

            # 总Q损失 (0.1*individual + 0.9*composite + 0.1*balance)
            qf_loss = (0.1 * (qf1_energy_loss + qf2_energy_loss +
                             qf1_carbon_loss + qf2_carbon_loss +
                             qf1_penalty_loss + qf2_penalty_loss) +
                      0.9 * (qf1_composite_loss + qf2_composite_loss) +
                      0.1 * usage_balance_loss)

            # 优化Q网络
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # 更新策略网络 (每步更新，与DDPG基线一致)
            actions_pred = actor(data.observations)

            # 获取双Q网络的Q值
            qf1_energy_pi, qf1_carbon_pi, qf1_penalty_pi, _, _, _ = qf1(data.observations, actions_pred, training=False)
            qf2_energy_pi, qf2_carbon_pi, qf2_penalty_pi, _, _, _ = qf2(data.observations, actions_pred, training=False)

            # 使用双Q网络的平均值
            avg_energy_qf_pi = (qf1_energy_pi + qf2_energy_pi) / 2
            avg_carbon_qf_pi = (qf1_carbon_pi + qf2_carbon_pi) / 2
            avg_penalty_qf_pi = (qf1_penalty_pi + qf2_penalty_pi) / 2

            # 使用权重组合三个critic的输出
            weighted_qf_pi = (args.energy_weight * avg_energy_qf_pi +
                            args.carbon_weight * avg_carbon_qf_pi +
                            args.penalty_weight * avg_penalty_qf_pi)

            actor_loss = -weighted_qf_pi.mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # 软更新目标网络
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 1000 == 0:
                # 记录门控权重分布
                for i in range(qf1.encoder.num_experts):
                    writer.add_scalar(f"gating/expert_{i}_weight", qf1_energy_gate_weights[0, i].item(), global_step)

                writer.add_scalar("losses/usage_balance_loss", usage_balance_loss.item(), global_step)
                writer.add_scalar("losses/qf1_energy_values", qf1_energy.mean().item(), global_step)
                writer.add_scalar("losses/qf2_energy_values", qf2_energy.mean().item(), global_step)
                writer.add_scalar("losses/qf1_carbon_values", qf1_carbon.mean().item(), global_step)
                writer.add_scalar("losses/qf2_carbon_values", qf2_carbon.mean().item(), global_step)
                writer.add_scalar("losses/qf1_penalty_values", qf1_penalty.mean().item(), global_step)
                writer.add_scalar("losses/qf2_penalty_values", qf2_penalty.mean().item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

                # 记录专家使用率
                for i in range(qf1.encoder.num_experts):
                    writer.add_scalar(f"gating/expert_{i}_usage", qf1.encoder.energy_expert_usage[i].item(), global_step)

    # 保存最终模型
    model_path = f"models/{run_name}"
    os.makedirs(model_path, exist_ok=True)
    torch.save(actor.state_dict(), f"{model_path}/actor.pth")
    torch.save(qf1.state_dict(), f"{model_path}/qf1.pth")
    torch.save(qf2.state_dict(), f"{model_path}/qf2.pth")
    print(f"Models saved to {model_path}")
    
    if best_model_path is not None:
        print(f"Best model was saved at: {best_model_path}")
        print(f"Best evaluation return: {best_episode_return:.3f}")

    envs.close()
    eval_env.close()
    writer.close()
    print("Training completed!")
