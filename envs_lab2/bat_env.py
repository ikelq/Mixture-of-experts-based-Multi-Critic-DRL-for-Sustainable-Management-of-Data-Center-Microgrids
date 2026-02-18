import numpy as np
import gymnasium as gym
import pandas as pd
import os

file_path = os.path.abspath(__file__)
PATH = os.path.split(os.path.dirname(file_path))[0]

class BatteryEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()

        self.bat_energy_cap = 4
        self.bat_power_cap = 1
        self.bat_eff = 0.98
        
        PV = pd.read_csv(PATH+f"/data/CarbonIntensity/uk_data.csv")['SOLAR'].values[:8760].reshape(365,24).astype(float)
        PV_capacity = 3   # 太阳电池容量MW
        self.PV_norm = PV_capacity * (PV - PV.min()) / (PV.max() - PV.min())

        # observation_space与reset/step返回shape一致，假设为[time, SoC]，如有更多变量请补充
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 2.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
       
        # 修改为连续action空间，范围[-1,1]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32), 
            high=np.array([1.0], dtype=np.float32), 
            shape=(1,), 
            dtype=np.float32
        )
        
    def reset(self, seed=None, day_id=0):
        self.day_id = day_id
        self.step_id = 0

        self.PV_day_hour = self.PV_norm[self.day_id]
        PV_gen = self.PV_day_hour[self.step_id].item()

        self.bat_soc = 0.2
        state = np.array([self.bat_soc, PV_gen], dtype=np.float32)
        info = {"bat_soc": self.bat_soc, "PV_gen": PV_gen}
        return state, info

    def step(self, action):
        # 确保action是标量
        action = float(action)
        
        # 计算实际充放电功率
        min_action = -min(self.bat_power_cap, 
                          self.bat_soc*self.bat_energy_cap*self.bat_eff)
        
        max_action = min(self.bat_power_cap, 
                         (1-self.bat_soc)*self.bat_energy_cap/self.bat_eff)
                         
        action_real = (action+1) * (max_action - min_action) / 2 + min_action
        # action_real = 0
        # 更新电池状态
        if action_real > 0:
            self.bat_soc += action_real*self.bat_eff/self.bat_energy_cap
        else:
            self.bat_soc += action_real/self.bat_eff/self.bat_energy_cap

        # 更新 step_id 并获取 PV_gen
        self.step_id += 1
        
        # 检查是否超出边界
        if self.step_id >= len(self.PV_day_hour):
            done = True
            truncated = False
            PV_gen = float(self.PV_day_hour[-1])  # 使用最后一个值
        else:
            done = False
            truncated = False
            PV_gen = float(self.PV_day_hour[self.step_id])

        # 计算奖励（示例：鼓励保持电池在50%容量左右）
        # target_soc = self.battery.capacity * 0.5
        reward = - (action_real - PV_gen)
        
        # 更新状态
        state = np.array([self.bat_soc, PV_gen], dtype=np.float32)
        
        info = {
            'bat_rl_action': action,
            'charge_MW': action_real,  
            'max_charge_MW': max_action,
            'min_charge_MW': min_action,
            'current_soc': self.bat_soc,
            'PV_gen': PV_gen
        }
        
        return state, reward, done, truncated, info

if __name__ == "__main__":
    # 创建环境
    env = BatteryEnv()
    
    # 重置环境
    obs, info = env.reset()
    info_list = []
    
    # 运行96步
    for step in range(24):
        # 执行动作（这里使用固定动作0.5作为示例）
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
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
    df.to_csv('info_battery_env.csv', index=False)
    print("\nInfo data saved to battery_env_info.csv")
    
    # 关闭环境
    env.close()

