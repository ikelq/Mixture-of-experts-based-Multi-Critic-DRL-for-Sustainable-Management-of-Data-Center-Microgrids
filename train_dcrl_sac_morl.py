#!/usr/bin/env python3
"""
SAC训练DCRL环境的脚本 - MORL动态权重版本（简化版）
基于cleanRL的SAC实现，适配DCRL环境
MORL扩展版本：支持用户偏好适应
- 简单MLP网络结构（无MoE）
- 三个任务特定Q函数：Energy, Carbon, Penalty
- 动态权重：Energy和Carbon使用动态权重w和(1-w)，Penalty固定
- 训练时随机采样权重w∈[0,1]，作为额外输入
- 测试时用户指定偏好权重，无需重新训练
- 可生成Pareto前沿展示energy-carbon trade-off
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

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
    exp_name: str = "dcrl_sac_morl"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "dcrl-sac-morl"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

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
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 2400
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    alpha_lr: float = 3e-4
    """the learning rate of the entropy coefficient optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    # Evaluation arguments
    eval_frequency: int = 2400
    """frequency of evaluation episodes"""
    eval_episodes: list = range(0, 360, 10)
    """evaluation episode day IDs"""

    # MORL specific arguments
    eval_weights: str = "0.0,0.25,0.5,0.75,1.0"
    """comma-separated list of energy weights to evaluate (carbon weight = 1 - energy_weight)"""
    penalty_weight: float = 1.0
    """fixed weight for penalty critic (constraint, not traded off)"""


class MultiRewardReplayBuffer:
    """
    自定义ReplayBuffer，用于存储多个sub-rewards
    """
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
        
        self.observations      = np.zeros((buffer_size, n_envs) + obs_shape, dtype=np.float32)
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
        """
        添加数据到buffer
        从infos中提取sub-rewards
        """
        # 复制数据以避免引用修改
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(actions).copy()
        self.dones[self.pos] = np.array(dones).copy()
        
        # 从infos中提取sub-rewards
        energy_rewards = np.zeros(self.n_envs)
        carbon_rewards = np.zeros(self.n_envs)
        penalty_rewards = np.zeros(self.n_envs)

        # Handle gym.vector.SyncVectorEnv info structure
        if isinstance(infos, dict):
            if 'energy_reward' in infos:
                energy_rewards = np.array(infos['energy_reward']).flatten()
                carbon_rewards = np.array(infos['carbon_reward']).flatten()
                penalty_rewards = np.array(infos['workload_reward']).flatten()
            elif 'final_info' in infos:
                for i in range(self.n_envs):
                    if infos['final_info'][i] is not None:
                        final_info = infos['final_info'][i]
                        energy_rewards[i] = final_info.get('energy_reward', 0)
                        carbon_rewards[i] = final_info.get('carbon_reward', 0)
                        penalty_rewards[i] = final_info.get('workload_reward', 0)
        else:
            for i in range(self.n_envs):
                info = infos[i] if i < len(infos) and infos[i] is not None else {}
                energy_rewards[i] = info.get('energy_reward', 0)
                carbon_rewards[i] = info.get('carbon_reward', 0)
                penalty_rewards[i] = info.get('workload_reward', 0)
        
        self.energy_rewards[self.pos] = energy_rewards
        self.carbon_rewards[self.pos] = carbon_rewards
        self.penalty_rewards[self.pos] = penalty_rewards
        
        # 计算总reward
        total_rewards = energy_rewards + carbon_rewards + penalty_rewards
        self.total_rewards[self.pos] = total_rewards
        
        # 更新位置
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True
    
    def sample(self, batch_size):
        """
        采样batch数据
        """
        if self.full:
            indices = np.random.randint(0, self.buffer_size, size=batch_size)
        else:
            indices = np.random.randint(0, self.pos, size=batch_size)
        
        # 随机选择环境索引
        env_indices = np.random.randint(0, self.n_envs, size=batch_size)
        
        batch_obs = torch.FloatTensor(self.observations[indices, env_indices]).to(self.device)
        batch_next_obs = torch.FloatTensor(self.next_observations[indices, env_indices]).to(self.device)
        batch_actions = torch.FloatTensor(self.actions[indices, env_indices]).to(self.device)
        batch_rewards = torch.FloatTensor(self.total_rewards[indices, env_indices]).to(self.device)
        batch_dones = torch.FloatTensor(self.dones[indices, env_indices]).to(self.device)
        
        # 也返回sub-rewards
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

        # 添加episode统计
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
        
        # 添加episode统计到info
        if terminated or truncated:
            info['episode'] = {
                'r': self.episode_return,
                'l': self.episode_length
            }
            
        return obs, reward, terminated, truncated, info


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    """Actor network with weight conditioning for MORL.
    
    The actor takes both observation and energy weight w as input.
    w=1 means only optimize energy, w=0 means only optimize carbon.
    """
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)

        # +1 for the weight input
        self.fc1 = nn.Linear(obs_dim + 1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)

    def forward(self, x, weight):
        """Forward pass with weight conditioning.
        
        Args:
            x: observation tensor [batch, obs_dim]
            weight: energy weight tensor [batch, 1], values in [0, 1]
        """
        # Concatenate observation with weight
        x_w = torch.cat([x, weight], dim=-1)
        x_w = F.relu(self.fc1(x_w))
        x_w = F.relu(self.fc2(x_w))
        mean = self.fc_mean(x_w)
        log_std = self.fc_logstd(x_w)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    def get_action(self, x, weight, deterministic=False):
        """Get action with weight conditioning.
        
        Args:
            x: observation tensor [batch, obs_dim]
            weight: energy weight tensor [batch, 1], values in [0, 1]
            deterministic: if True, return mean action (for evaluation)
        """
        mean, log_std = self(x, weight)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        if deterministic:
            x_t = mean
        else:
            x_t = normal.rsample()

        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean


class QNetwork(nn.Module):
    """Q network with weight conditioning for MORL.
    
    Simple MLP architecture with three separate output heads for:
    - Energy Q-value
    - Carbon Q-value  
    - Penalty Q-value
    """
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        
        # +1 for weight input
        input_dim = obs_dim + action_dim + 1
        
        # Shared layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Task-specific heads
        self.energy_head = nn.Linear(256, 1)
        self.carbon_head = nn.Linear(256, 1)
        self.penalty_head = nn.Linear(256, 1)

    def forward(self, x, a, w):
        """Forward pass with weight conditioning.
        
        Args:
            x: observation tensor [batch, obs_dim]
            a: action tensor [batch, action_dim]
            w: weight tensor [batch, 1]
            
        Returns:
            energy_q, carbon_q, penalty_q: Q-values for each objective
        """
        # Concatenate inputs
        x_a_w = torch.cat([x, a, w], dim=1)
        
        # Shared features
        features = F.relu(self.fc1(x_a_w))
        features = F.relu(self.fc2(features))
        
        # Task-specific Q-values
        energy_q = self.energy_head(features)
        carbon_q = self.carbon_head(features)
        penalty_q = self.penalty_head(features)
        
        return energy_q, carbon_q, penalty_q


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
    print(f"Using device: {device}")

    # env setup
    def make_env():
        def _init():
            return DCRLWrapper()
        return _init

    envs = gym.vector.SyncVectorEnv(
        [make_env() for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Evaluation environment setup
    eval_env = DCRLWrapper()

    print(f"Observation space: {envs.single_observation_space}")
    print(f"Action space: {envs.single_action_space}")

    # Parse evaluation weights
    eval_weights_list = [float(w) for w in args.eval_weights.split(',')]
    print(f"MORL evaluation weights (energy): {eval_weights_list}")

    actor = Actor(envs).to(device)

    # 创建Q网络（每个都有双Q网络）
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    # 优化器
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()),
        lr=args.q_lr
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)
    else:
        alpha = args.alpha

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

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    print("Starting training...")
    
    best_episode_return = float('-inf')
    
    # 添加sub-reward统计
    energy_reward_sum = 0.0
    carbon_reward_sum = 0.0
    penalty_reward_sum = 0.0
    step_count = 0

    # MORL: Current weight for the episode (randomly sampled at episode start)
    current_weight = np.random.uniform(0, 1)
    current_weight_tensor = torch.FloatTensor([[current_weight]]).to(device)

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            # MORL: Pass weight to actor
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device), current_weight_tensor.expand(args.num_envs, -1))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # MORL: Resample weight at episode end
        if any(terminations) or any(truncations):
            current_weight = np.random.uniform(0, 1)
            current_weight_tensor = torch.FloatTensor([[current_weight]]).to(device)

        # Evaluation logic - MORL: evaluate with multiple weights
        if global_step % args.eval_frequency == 0:
            # Store w=0.5 results for compatibility logging
            w05_return = None
            w05_length = None

            # Evaluate with each weight in eval_weights_list
            for eval_w in eval_weights_list:
                eval_weight_tensor = torch.FloatTensor([[eval_w]]).to(device)
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
                        # Use deterministic actions for evaluation
                        with torch.no_grad():
                            action, _, _ = actor.get_action(
                                torch.Tensor(eval_obs).unsqueeze(0).to(device),
                                eval_weight_tensor,
                                deterministic=True
                            )
                            action = action.squeeze(0).cpu().numpy()

                        eval_obs, reward, terminated, truncated, info = eval_env.step(action)
                        episode_return += reward
                        episode_length += 1

                        # Collect sub-rewards
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

                # Log evaluation metrics for this weight
                avg_eval_return = np.mean(eval_returns)
                avg_eval_length = np.mean(eval_lengths)
                avg_eval_energy = np.mean(eval_energy_rewards)
                avg_eval_carbon = np.mean(eval_carbon_rewards)
                avg_eval_penalty = np.mean(eval_penalty_rewards)

                # Store w=0.5 results
                if eval_w == 0.5:
                    w05_return = avg_eval_return
                    w05_length = avg_eval_length

                weight_str = f"{eval_w:.2f}"
                print(f"Step {global_step:6d} | w={weight_str} | Return: {avg_eval_return:8.3f} | Energy: {avg_eval_energy:8.3f} | Carbon: {avg_eval_carbon:8.3f}")

                if eval_w == 0.5 and avg_eval_return > best_episode_return:
                    best_episode_return = avg_eval_return
                    print(f"New best evaluation return (w=0.5): {best_episode_return:.3f}")
                    # Save best model
                    best_model_path = f"models/{run_name}/best_model"
                    os.makedirs(best_model_path, exist_ok=True)
                    torch.save(actor.state_dict(), f"{best_model_path}/actor.pth")
                    torch.save(qf1.state_dict(), f"{best_model_path}/qf1.pth")
                    torch.save(qf2.state_dict(), f"{best_model_path}/qf2.pth")
                    print(f"New best model saved to {best_model_path}")

                writer.add_scalar(f"morl/eval_return_w{weight_str}", avg_eval_return, global_step)
                writer.add_scalar(f"morl/eval_energy_w{weight_str}", avg_eval_energy, global_step)
                writer.add_scalar(f"morl/eval_carbon_w{weight_str}", avg_eval_carbon, global_step)
                writer.add_scalar(f"morl/eval_penalty_w{weight_str}", avg_eval_penalty, global_step)

            # Log w=0.5 metrics with standard names for compatibility
            if w05_return is not None:
                writer.add_scalar("charts/eval_episodic_return", w05_return, global_step)
                writer.add_scalar("charts/eval_episodic_length", w05_length, global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`        
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        
        # 使用自定义ReplayBuffer
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # 统计sub-rewards（用于监控）
        if isinstance(infos, dict):
            if 'energy_reward' in infos:
                energy_reward_sum += np.sum(infos['energy_reward'])
                carbon_reward_sum += np.sum(infos['carbon_reward'])
                penalty_reward_sum += np.sum(infos['workload_reward'])
                step_count += args.num_envs
            elif 'final_info' in infos:
                for i in range(args.num_envs):
                    if infos['final_info'][i] is not None:
                        final_info = infos['final_info'][i]
                        energy_reward_sum += final_info.get('energy_reward', 0)
                        carbon_reward_sum += final_info.get('carbon_reward', 0)
                        penalty_reward_sum += final_info.get('workload_reward', 0)
                        step_count += 1
        else:
            for info in infos:
                if info is not None:
                    energy_reward_sum += info.get('energy_reward', 0)
                    carbon_reward_sum += info.get('carbon_reward', 0)
                    penalty_reward_sum += info.get('workload_reward', 0)
                    step_count += 1

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            # MORL: Sample random weights for this batch
            batch_weights = torch.rand(args.batch_size, 1).to(device)  # w ~ U(0,1)

            with torch.no_grad():
                # MORL: Pass weights to actor
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations, batch_weights)

                # Get Q-values from target networks
                qf1_next_target_energy, qf1_next_target_carbon, qf1_next_target_penalty = qf1_target(data.next_observations, next_state_actions, batch_weights)
                qf2_next_target_energy, qf2_next_target_carbon, qf2_next_target_penalty = qf2_target(data.next_observations, next_state_actions, batch_weights)

                # Compute minimum Q-values for each task
                min_qf_next_target_energy = torch.min(qf1_next_target_energy, qf2_next_target_energy)
                min_qf_next_target_carbon = torch.min(qf1_next_target_carbon, qf2_next_target_carbon)
                min_qf_next_target_penalty = torch.min(qf1_next_target_penalty, qf2_next_target_penalty)

                # Compute target Q-values (WITHOUT entropy in individual Q-functions)
                # In multi-objective RL, each sub-Q learns the value of its own reward stream
                # Entropy regularization is only applied in the actor loss
                # This avoids the problem of entropy being counted multiple times
                next_energy_q_value = data.energy_rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next_target_energy.view(-1)
                next_carbon_q_value = data.carbon_rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next_target_carbon.view(-1)
                next_penalty_q_value = data.penalty_rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next_target_penalty.view(-1)

            # Get current Q-values
            qf1_energy, qf1_carbon, qf1_penalty = qf1(data.observations, data.actions, batch_weights)
            qf2_energy, qf2_carbon, qf2_penalty = qf2(data.observations, data.actions, batch_weights)
            
            # Compute losses for each task
            qf1_energy_loss =  F.mse_loss(qf1_energy.view(-1), next_energy_q_value)
            qf2_energy_loss =  F.mse_loss(qf2_energy.view(-1), next_energy_q_value)
            qf1_carbon_loss =  F.mse_loss(qf1_carbon.view(-1), next_carbon_q_value)     
            qf2_carbon_loss =  F.mse_loss(qf2_carbon.view(-1), next_carbon_q_value)
            qf1_penalty_loss = F.mse_loss(qf1_penalty.view(-1), next_penalty_q_value)
            qf2_penalty_loss = F.mse_loss(qf2_penalty.view(-1), next_penalty_q_value)
            
            # Total Q loss
            qf_loss = (qf1_energy_loss + qf2_energy_loss + 
                       qf1_carbon_loss + qf2_carbon_loss + 
                       qf1_penalty_loss + qf2_penalty_loss)

            # Optimize Q networks
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    # MORL: Sample new weights for policy update
                    policy_weights = torch.rand(args.batch_size, 1).to(device)  # w ~ U(0,1)

                    # MORL: Pass weights to actor
                    pi, log_pi, _ = actor.get_action(data.observations, policy_weights)

                    # Get Q-values
                    qf1_energy_pi, qf1_carbon_pi, qf1_penalty_pi = qf1(data.observations, pi, policy_weights)
                    qf2_energy_pi, qf2_carbon_pi, qf2_penalty_pi = qf2(data.observations, pi, policy_weights)

                    # Compute minimum Q-values for each task
                    min_energy_qf_pi = torch.min(qf1_energy_pi, qf2_energy_pi)
                    min_carbon_qf_pi = torch.min(qf1_carbon_pi, qf2_carbon_pi)
                    min_penalty_qf_pi = torch.min(qf1_penalty_pi, qf2_penalty_pi)

                    # MORL: Use dynamic weights for energy and carbon, fixed weight for penalty
                    # w * Q_energy + (1-w) * Q_carbon + penalty_weight * Q_penalty
                    weighted_qf_pi = (policy_weights * min_energy_qf_pi +
                                    (1 - policy_weights) * min_carbon_qf_pi +
                                    args.penalty_weight * min_penalty_qf_pi)

                    actor_loss = ((alpha * log_pi) - weighted_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations, policy_weights)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 1000 == 0:
                writer.add_scalar("losses/qf1_energy_values", qf1_energy.mean().item(), global_step)
                writer.add_scalar("losses/qf2_energy_values", qf2_energy.mean().item(), global_step)
                writer.add_scalar("losses/qf1_carbon_values", qf1_carbon.mean().item(), global_step)
                writer.add_scalar("losses/qf2_carbon_values", qf2_carbon.mean().item(), global_step)
                writer.add_scalar("losses/qf1_penalty_values", qf1_penalty.mean().item(), global_step)
                writer.add_scalar("losses/qf2_penalty_values", qf2_penalty.mean().item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                
                # 记录sub-rewards统计
                if step_count > 0:
                    writer.add_scalar("sub_rewards/energy_avg", energy_reward_sum / step_count, global_step)
                    writer.add_scalar("sub_rewards/carbon_avg", carbon_reward_sum / step_count, global_step)
                    writer.add_scalar("sub_rewards/penalty_avg", penalty_reward_sum / step_count, global_step)
                    
                    # 从采样的batch中记录sub-rewards分布
                    writer.add_scalar("batch_sub_rewards/energy_mean", data.energy_rewards.mean().item(), global_step)
                    writer.add_scalar("batch_sub_rewards/carbon_mean", data.carbon_rewards.mean().item(), global_step)
                    writer.add_scalar("batch_sub_rewards/penalty_mean", data.penalty_rewards.mean().item(), global_step)
                
                print(f"Step {global_step:6d} | SPS: {int(global_step / (time.time() - start_time)):4d}")
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    # 保存模型
    model_path = f"models/{run_name}"
    os.makedirs(model_path, exist_ok=True)
    torch.save(actor.state_dict(), f"{model_path}/actor.pth")
    torch.save(qf1.state_dict(), f"{model_path}/qf1.pth")
    torch.save(qf2.state_dict(), f"{model_path}/qf2.pth")
    print(f"Models saved to {model_path}")

    envs.close()
    eval_env.close()
    writer.close()
    print("Training completed!")
