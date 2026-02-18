import os
from typing import Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd

from envs_lab2.workload_env import CarbonLoadEnv
from envs_lab2.bat_env import BatteryEnv
from envs_lab2.dc_env import dc_gymenv

file_path = os.path.abspath(__file__)
PATH = os.path.dirname(file_path)

print(f"PATH: {PATH}")

def sc_time(step_id):
    step_id_cos_hour = np.cos(step_id/12)*0.5 + 0.5
    step_id_sin_hour = np.sin(step_id/12)*0.5 + 0.5
    return [step_id_cos_hour, step_id_sin_hour]


class DCRL(gym.Env):
    def __init__(self):
        super().__init__()

        # 初始化工作负载环境
        self.workload_env = CarbonLoadEnv()
        # 初始化电池环境
        self.battery_env = BatteryEnv()
        # 初始化数据中心环境
        self.dc_env = dc_gymenv()

        carbon_data_list = pd.read_csv(PATH+f"/data/CarbonIntensity/uk_data.csv")['CARBON_INTENSITY'].values[:8760]
        self.carbon_data = carbon_data_list.reshape(365,24).astype(float)
        self.carbon_data_norm = 2*self.carbon_data  / self.carbon_data.max() 

        # Adding a time dimension to the observation space
        low_time = np.array([0.0, 0.0], dtype=np.float32)
        high_time = np.array([1.0, 1.0], dtype=np.float32)
        
        # 确保所有边界值都是float32类型
        low = np.concatenate([
            low_time,
            np.zeros(24, dtype=np.float32),
            self.workload_env.observation_space.low.astype(np.float32),
            self.dc_env.observation_space.low.astype(np.float32),
            self.battery_env.observation_space.low.astype(np.float32)
        ])
        
        high = np.concatenate([
            high_time,
            2*np.ones(24, dtype=np.float32),
            self.workload_env.observation_space.high.astype(np.float32),
            self.dc_env.observation_space.high.astype(np.float32),
            self.battery_env.observation_space.high.astype(np.float32)
        ])
        
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )

        # 合并两个子环境的动作空间为一个Box空间
        low = np.concatenate([
            self.workload_env.action_space.low.astype(np.float32),
            self.dc_env.action_space.low.astype(np.float32),
            self.battery_env.action_space.low.astype(np.float32)            
        ])
        high = np.concatenate([
            self.workload_env.action_space.high.astype(np.float32),
             self.dc_env.action_space.high.astype(np.float32),
            self.battery_env.action_space.high.astype(np.float32)      
        ])
        self.action_space = gym.spaces.Box(
            low=low, 
              high=high,
            dtype=np.float32
        )
        
    def reset(self, day_id = None, seed=None, options=None):
        """
        Reset the environment.

        Returns:
            obs (np.ndarray): Initial observation.
            info (dict): Additional info.
        """

        if seed is not None:
            np.random.seed(seed)
            self.workload_env.action_space.seed(seed)
            # 可根据需要为其他子环境设置 seed

        if day_id is None:
            self.day_id = np.random.randint(0,365)
        else:
            self.day_id = day_id
        self.carbon_data_day = self.carbon_data[self.day_id]
        self.carbon_data_day_norm = self.carbon_data_norm[self.day_id]
        # self.carbon_data_day_norm = (self.carbon_data_day_norm -self.carbon_data_day_norm.min()) / (self.carbon_data_day_norm.max() - self.carbon_data_day_norm.min())

        self.step_id = 0

        workload_state, workload_info = self.workload_env.reset(self.day_id)
        dc_state, dc_info = self.dc_env.reset(self.day_id, workload_info['cpu_load'])
        battery_state, battery_info = self.battery_env.reset()

        obs = np.hstack((
            sc_time(self.step_id),    
            self.carbon_data_day_norm,
            workload_state,       
            dc_state,
            battery_state,
        )).astype(np.float32)
        # info_dict = {**workload_info, **battery_info, **dc_info}
        info_dict = {}
        return obs, info_dict

    def step(self, action):
        # 拆分 action
        action_workload, action_dc, action_battery  = action
        workload_state, workload_reward, workload_done, workload_truncated, workload_info = self.workload_env.step(action_workload)
        cpu_load = workload_info['cpu_load']
        dc_state, dc_reward, dc_done, dc_truncated, dc_info = self.dc_env.step(action_dc,cpu_load)
        battery_state, battery_reward, battery_done, battery_truncated, battery_info = self.battery_env.step(action_battery)
 
        obs = np.hstack((
            sc_time(self.step_id),           # 时间
            self.carbon_data_day_norm,
            workload_state, 
            dc_state,  
            battery_state,   
        )).astype(np.float32)

        info_dict = {**workload_info, **battery_info, **dc_info}
 
        dc_carbon_reward = -info_dict['dc_total_power_MW'] * self.carbon_data_day_norm[self.step_id]
        battery_carbon_reward = -(info_dict['charge_MW'] - info_dict['PV_gen']) * self.carbon_data_day_norm[self.step_id]
        
        reward = float(workload_reward + dc_carbon_reward + battery_carbon_reward + dc_reward + battery_reward)

        add_info = {"step_id": self.step_id, "ci": self.carbon_data_day[self.step_id], "ci_norm": self.carbon_data_day_norm[self.step_id], "workload_reward": workload_reward, "dc_carbon_reward": dc_carbon_reward, "battery_carbon_reward": battery_carbon_reward, "dc_reward": dc_reward, "battery_reward": battery_reward, "carbon_reward": dc_carbon_reward+battery_carbon_reward,'energy_reward':dc_reward+battery_reward, "reward": reward}
        info_all = {**info_dict, **add_info}

        self.step_id += 1

        terminated = workload_done
        truncated = False

        # print(f"obs: {obs}", f"reward: {reward}")

        return obs, reward, terminated, truncated, info_all

if __name__ == '__main__':

    env = DCRL()
    np.random.seed(42)
    env.action_space.seed(42)
    obs, info = env.reset(seed=42)
    step = 0
    total_reward = 0
    info_all_list = []

    done = False

    while not done:
        # action = env.action_space.sample() 
        action = np.array([1, 0, 0])
        # action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated
        total_reward += reward
        step += 1
        info_all_list.append(info)

    print("End of episode.")
    print("="*60)
    print(f"Episode Summary (Steps: {step})")
    print("="*60)
    print(f"Total episode reward = {total_reward:.4f}")
    print("="*60)
    df = pd.DataFrame(info_all_list)
    df.to_csv('info_all_data.csv', index=False)
    print("Info All List saved to info_all_data.csv")