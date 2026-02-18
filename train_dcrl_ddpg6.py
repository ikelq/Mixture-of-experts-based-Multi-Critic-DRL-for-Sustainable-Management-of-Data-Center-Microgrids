#!/usr/bin/env python3
"""
DDPG训练DCRL环境的脚本
基于cleanRL的DDPG实现，适配DCRL环境
- 确定性策略网络
- 单一Q网络
- 标准DDPG流程
- 基于train_dcrl_sac6.py修改
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
    exp_name: str = "dcrl_ddpg6"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "dcrl-ddpg"
    wandb_entity: str = None
    capture_video: bool = False
    total_timesteps: int = int(1e6)
    num_envs: int = 1
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 2400
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    target_network_frequency: int = 1
    eval_frequency: int = 2400
    eval_episodes: list = range(0, 360, 10)
    noise_scale: float = 0.1  # 探索噪声的缩放因子

class ReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device, n_envs=1):
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
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.pos = 0
        self.full = False
    def add(self, obs, next_obs, actions, rewards, dones, infos):
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(actions).copy()
        self.rewards[self.pos] = np.array(rewards).copy()
        self.dones[self.pos] = np.array(dones).copy()
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True
    def sample(self, batch_size):
        if self.full:
            indices = np.random.randint(0, self.buffer_size, size=batch_size)
        else:
            indices = np.random.randint(0, self.pos, size=batch_size)
        env_indices = np.random.randint(0, self.n_envs, size=batch_size)
        batch_obs = torch.FloatTensor(self.observations[indices, env_indices]).to(self.device)
        batch_next_obs = torch.FloatTensor(self.next_observations[indices, env_indices]).to(self.device)
        batch_actions = torch.FloatTensor(self.actions[indices, env_indices]).to(self.device)
        batch_rewards = torch.FloatTensor(self.rewards[indices, env_indices]).to(self.device)
        batch_dones = torch.FloatTensor(self.dones[indices, env_indices]).to(self.device)
        class ReplayBufferSamples:
            def __init__(self, observations, next_observations, actions, rewards, dones):
                self.observations = observations
                self.next_observations = next_observations
                self.actions = actions
                self.rewards = rewards
                self.dones = dones
        return ReplayBufferSamples(batch_obs, batch_next_obs, batch_actions, batch_rewards, batch_dones)

class DCRLWrapper(gym.Env):
    def __init__(self):
        self.env = DCRL()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.episode_return = 0
        self.episode_length = 0
    def reset(self, day_id = None, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            self.env.workload_env.action_space.seed(seed)
            self.env.dc_env.action_space.seed(seed)
            self.env.battery_env.action_space.seed(seed)
        self.episode_return = 0
        self.episode_length = 0
        return self.env.reset(day_id = day_id)
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

class DDPGActor(nn.Module):
    """DDPG的确定性策略网络"""
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # 输出范围限制在[-1, 1]
        )
        
    def forward(self, x):
        return self.net(x)
    
    def get_action(self, x, deterministic=False):
        action = self(x)
        if not deterministic:
            # 添加探索噪声
            noise = torch.randn_like(action) * 0.1
            action = action + noise
            action = torch.clamp(action, -1, 1)  # 确保动作在有效范围内
        return action

class DDPGQNetwork(nn.Module):
    """DDPG的Q网络"""
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x, a):
        x_a = torch.cat([x, a], 1)
        return self.net(x_a)

if __name__ == "__main__":
    import stable_baselines3 as sb3
    if sb3.__version__ < "2.0":
        print("Warning: This script is optimized for stable_baselines3>=2.0")
    args = tyro.cli(Args)
    run_name = f"dcrl__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    envs = gym.vector.SyncVectorEnv(
        [lambda: DCRLWrapper() for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    eval_env = DCRLWrapper()
    print(f"Observation space: {envs.single_observation_space}")
    print(f"Action space: {envs.single_action_space}")
    
    # 创建网络
    actor = DDPGActor(envs).to(device)
    actor_target = DDPGActor(envs).to(device)
    actor_target.load_state_dict(actor.state_dict())
    
    q_network = DDPGQNetwork(envs).to(device)
    q_target = DDPGQNetwork(envs).to(device)
    q_target.load_state_dict(q_network.state_dict())
    
    # 优化器
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)
    q_optimizer = optim.Adam(q_network.parameters(), lr=args.q_lr)
    
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
    )
    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    print("Starting training...")
    best_episode_return = float('-inf')
    best_model_path = None
    
    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()
        
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        if global_step % args.eval_frequency == 0:
            eval_returns = []
            eval_lengths = []
            eval_energy_rewards = []
            eval_carbon_rewards = []
            eval_penalty_rewards = []
            for day_id in args.eval_episodes:
                eval_obs, _ = eval_env.reset(day_id = day_id, seed=args.seed)
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
            avg_eval_return = np.mean(eval_returns)
            avg_eval_length = np.mean(eval_lengths)
            avg_eval_energy = np.mean(eval_energy_rewards)
            avg_eval_carbon = np.mean(eval_carbon_rewards)
            avg_eval_penalty = np.mean(eval_penalty_rewards)
            print(f"Step {global_step:6d} | Eval Return: {avg_eval_return:8.3f} | Eval Length: {int(avg_eval_length):3d}")
            if avg_eval_return > best_episode_return:
                best_episode_return = avg_eval_return
                print(f"New best evaluation return: {best_episode_return:.3f}")
                best_model_path = f"models/{run_name}"
                os.makedirs(best_model_path, exist_ok=True)
                torch.save(actor.state_dict(), f"{best_model_path}/actor.pth")
                torch.save(q_network.state_dict(), f"{best_model_path}/q_network.pth")
                print(f"New best model saved to {best_model_path}")
            writer.add_scalar("charts/eval_episodic_return", avg_eval_return, global_step)
            writer.add_scalar("charts/eval_episodic_length", avg_eval_length, global_step)
            writer.add_scalar("eval_sub_rewards/energy_avg", avg_eval_energy, global_step)
            writer.add_scalar("eval_sub_rewards/carbon_avg", avg_eval_carbon, global_step)
            writer.add_scalar("eval_sub_rewards/penalty_avg", avg_eval_penalty, global_step)
        
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs
        
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            
            # 计算目标Q值
            with torch.no_grad():
                next_actions = actor_target(data.next_observations)
                target_q = q_target(data.next_observations, next_actions)
                target_q = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * target_q.view(-1)
            
            # 更新Q网络
            current_q = q_network(data.observations, data.actions)
            q_loss = F.mse_loss(current_q.view(-1), target_q)
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()
            
            # 更新策略网络

            actions_pred = actor(data.observations)
            actor_loss = -q_network(data.observations, actions_pred).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            
            # 软更新目标网络
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(q_network.parameters(), q_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            
            if global_step % 1000 == 0:
                writer.add_scalar("losses/q_values", current_q.mean().item(), global_step)
                writer.add_scalar("losses/q_loss", q_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
    
    model_path = f"models/{run_name}"
    os.makedirs(model_path, exist_ok=True)
    torch.save(actor.state_dict(), f"{model_path}/actor.pth")
    torch.save(q_network.state_dict(), f"{model_path}/q_network.pth")
    print(f"Models saved to {model_path}")
    
    # 保存最佳模型（如果存在）
    if best_model_path is not None:
        print(f"Best model was saved at: {best_model_path}")
        print(f"Best evaluation return: {best_episode_return:.3f}")
    else:
        print("No best model was saved during training")

    envs.close()
    eval_env.close()
    writer.close()
    print("Training completed!") 