from typing import Optional, Tuple
import numpy as np
import pandas as pd
import os
import json

import gymnasium as gym

from .IT_HVAC import IT_HVAC

file_path = os.path.abspath(__file__)
PATH = os.path.split(os.path.dirname(file_path))[0]

class dc_gymenv(gym.Env):
    """
    A gymnasium environment for the data center cooling system.
    """
    def __init__(self):
        super().__init__()

        self.min_crac_temp = 15.0
        self.max_crac_temp = 22.0
        # 最大crac temp delta 调控量
        self.temp_delta_cap = 1

        # 获天气数据 obtain other weather files: https://climate.onebuilding.org/ 
        temperature_data = pd.read_csv(PATH+f"/data/CarbonIntensity/uk_data.csv")['temperature'].values[:8760]
        self.ambient_temp_day_hour = temperature_data.reshape(365,24).astype(float)
        self.time_steps = 24

        self.IT_HVAC = IT_HVAC()

        # 创建observation_space
        observation_variables = {
            'ambient_temp':[-10.0, 40.0],
            'CRAC_temp_setpoint':[10, 30],
            'cpu_load':[0.0, 1.0],
            'IT_power_W':[1e6,5e6],
            'HVAC_power_W':[1e5,1e6]
        }

        self.obs_low = np.array([var[0] for var in observation_variables.values()], dtype=np.float32)
        self.obs_high = np.array([var[1] for var in observation_variables.values()], dtype=np.float32)

        self.norm_obs_low = np.array([0 for _ in observation_variables.values()], dtype=np.float32)
        self.norm_obs_high = np.array([1 for _ in observation_variables.values()], dtype=np.float32)

        # self.obs_low = np.array([0 for _ in observation_variables.values()], dtype=np.float32)
        # self.obs_high = np.array([1 for _ in observation_variables.values()], dtype=np.float32)
        
        self.observation_space = gym.spaces.Box(
            low=self.norm_obs_low,
            high=self.norm_obs_high,
            dtype=np.float32
        )

        # 创建action_space
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,), dtype=np.float32)


    def reset(self, day_id=0, cpu_load=0.0):
        self.day_id = day_id
        self.step_id = 0
        self.cpu_load = cpu_load
        self.ambient_temp_day = self.ambient_temp_day_hour[self.day_id]
        
        ambient_temp = self.ambient_temp_day[self.step_id]
        self.CRAC_temp_setpoint = 16
        IT_power_consumption,  cpu_pwr,  P_cup_fan = self.IT_HVAC.IT_power_consumption(CRAC_temp_setpoint=self.CRAC_temp_setpoint, cpu_load=self.cpu_load)
        
        chiller_power_consumption = self.IT_HVAC.calculate_HVAC_power(CRAC_temp_setpoint=self.CRAC_temp_setpoint, ambient_temp=ambient_temp, IT_power_consumption=IT_power_consumption)

        HVAC_power_consumption = chiller_power_consumption

      # 初始化state
        state = np.asarray(np.hstack(([ambient_temp, self.CRAC_temp_setpoint, self.cpu_load, IT_power_consumption, HVAC_power_consumption])), dtype=np.float32)
        norm_state = (state - self.obs_low) / (self.obs_high - self.obs_low)
        
        info = {}
        return norm_state, info        

    def step(self, action, cpu_load):

        if isinstance(action, (np.ndarray, list)):
            action = action[0]

        self.cpu_load = cpu_load
        ambient_temp = self.ambient_temp_day[self.step_id]
        # ambient_temp = 30
        min_action = -min(self.temp_delta_cap,  self.CRAC_temp_setpoint - self.min_crac_temp)
        max_action = min(self.temp_delta_cap, self.max_crac_temp - self.CRAC_temp_setpoint)

        crac_setpoint_delta = (action+1) * (max_action - min_action) / 2 + min_action

        self.CRAC_temp_setpoint += crac_setpoint_delta

        IT_power_consumption,  cpu_pwr,  P_cup_fan = self.IT_HVAC.IT_power_consumption(CRAC_temp_setpoint=self.CRAC_temp_setpoint, cpu_load=self.cpu_load)      

        chiller_power_consumption = self.IT_HVAC.calculate_HVAC_power(CRAC_temp_setpoint=self.CRAC_temp_setpoint, ambient_temp=ambient_temp, IT_power_consumption=IT_power_consumption)
        HVAC_power_consumption = chiller_power_consumption
        # convert_power_to_MW = 1e6
        
        # calculate reward
        reward = -(IT_power_consumption + HVAC_power_consumption)/1e6

        # next step
        self.step_id += 1
        if self.step_id < self.time_steps:
            state = np.asarray(np.hstack(([self.ambient_temp_day[self.step_id], self.CRAC_temp_setpoint, self.cpu_load, IT_power_consumption, HVAC_power_consumption])), dtype=np.float32)
            done = False
            truncated = False
        else:
            state = np.asarray(np.hstack(([ambient_temp, self.CRAC_temp_setpoint, self.cpu_load, IT_power_consumption, HVAC_power_consumption])), dtype=np.float32)
            done = True
            truncated = False
        
        norm_state = (state - self.obs_low) / (self.obs_high - self.obs_low)

        # add info dictionary 
        self.info = {
            'dc_rl_action': action, #当前RL 动作
            'dc_env_action': crac_setpoint_delta, #当前crac setpoint变化
            'dc_crac_setpoint': self.CRAC_temp_setpoint, #当前crac setpoint
            'dc_ambient_temp': ambient_temp,
            # 'CRAC_return_temp': CRAC_return_temp, #当前CRAC return temp 
            'dc_IT_power_consumption_Mw': IT_power_consumption/1e6, #当前IT负荷
            'dc_HVAC_power_consumption_Mw': HVAC_power_consumption/1e6, #当前HVAC负荷
            'dc_total_power_MW': (IT_power_consumption + HVAC_power_consumption) / 1e6, #当前总负荷
            'dc_power_lb_W': 1.5e6,
            'dc_power_ub_W': 4e6,
            'dc_cpu_load': self.cpu_load, #当前cpu负荷
        }

        return norm_state, reward, done, truncated, self.info

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    
    # 创建环境
    env = dc_gymenv()
    
    # 重置环境
    obs, info = env.reset(day_id=200)
    info_list = []
    
    cpu_load = 0
    # 运行24步（模拟一天）
    action = np.array([0.5], dtype=np.float32)
    for step in range(24):
        # 执行动作（这里使用固定动作0.5作为示例）
        action = np.array([0.5], dtype=np.float32)

        cpu_load += 0.05
        obs, reward, done, truncated, info = env.step(action, cpu_load=cpu_load)
        
        # 收集info
        info_list.append(info)
        
        # 打印每一步的信息
        print(f"\nStep {step + 1}:")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print(f"Info: {info}")
        print("-" * 50)
        
        if done or truncated:
            break
    
    # 保存info为DataFrame
    df = pd.DataFrame(info_list)
    
    # 保存到CSV文件
    df.to_csv('info_dc_env.csv', index=False)
    print("\nInfo data saved to dc_env_info.csv")
    
    # 关闭环境
    env.close()

