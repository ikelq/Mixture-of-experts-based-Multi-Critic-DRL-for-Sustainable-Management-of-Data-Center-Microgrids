import os
import gymnasium as gym
import numpy as np
import pandas as pd

file_path = os.path.abspath(__file__)
PATH = os.path.split(os.path.dirname(file_path))[0]

class CarbonLoadEnv(gym.Env):
    def __init__(self):

        # carbon_data_list = pd.read_csv(PATH+f"/data/CarbonIntensity/NYIS_NG_&_avgCI.csv")['avg_CI'].values[:8760]
        # self.carbon_data_day_hour = carbon_data_list.reshape(365,24).astype(float)

        # Load 1-year CPU data from a CSV file - normalized [0,1]
        cpu_data = pd.read_csv(PATH+'/data/Workload/Alibaba_CPU_Data_Hourly_1.csv')['cpu_load'].values[:8760]
        # 计算每个小时的负荷
        self.cpu_data_day_hour = cpu_data.reshape(365,24).astype(float)
        flexible_workload_ratio=0.7
        self.cpu_data_day_hour_inflx = self.cpu_data_day_hour * (1-flexible_workload_ratio)
        self.cpu_data_day_hour_flx = self.cpu_data_day_hour - self.cpu_data_day_hour_inflx
        self.cpu_data_day_flx_sum = np.sum(self.cpu_data_day_hour_flx, axis=1)

        # self.action_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32), 
            high=np.array([1.0], dtype=np.float32), 
            shape=(1,), dtype=np.float32)
        
        low = np.concatenate([
            np.zeros(24, dtype=np.float32),
            np.zeros(24, dtype=np.float32),
            np.zeros(1, dtype=np.float32)
        ])
        high = np.concatenate([
            np.ones(24, dtype=np.float32),
            np.ones(24, dtype=np.float32),
            10*np.ones(1, dtype=np.float32)
        ])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.time_steps = 24
        self.future_steps=4

    def reset(self, day_id):
        
        # read day cpu data
        # self.day_id = 0
        self.day_id = day_id
        self.step_id = 0

        self.cpu_day_hour_inflx = self.cpu_data_day_hour_inflx[self.day_id]
        self.cpu_day_hour_flx = self.cpu_data_day_hour_flx[self.day_id]
        self.cpu_day_flx_sum = self.cpu_data_day_flx_sum[self.day_id]
        
        # self.queue = self.cpu_day_flx_sum
        self.queue = 0

        state = np.asarray(np.hstack(([self.cpu_day_hour_inflx, 
                                       self.cpu_day_hour_flx,
                                       self.queue
                                       ])), dtype=np.float32)
        
        cpu_load = self.cpu_day_hour_inflx[self.step_id]

        info = {"day": self.day_id, "cpu_day_hour_inflx": self.cpu_day_hour_inflx[self.step_id], "cpu_day_hour_flx": self.cpu_day_hour_flx[self.step_id], "cpu_load": cpu_load}
        
        return state, info

    def step(self, action):
        # action = action[0]
        
        # update queue
        self.queue += self.cpu_day_hour_flx[self.step_id]

        # 直接赋值，因为queue是标量值
        current_queue = float(self.queue)

        # max flexible workload that can be executed
        max_flx_execution = min(current_queue, 1 - self.cpu_day_hour_inflx[self.step_id])

        # execute workload
        executed_flx_workload = max_flx_execution * (action+1)/2

        # update queue
        self.queue -= executed_flx_workload

        # update total workload
        net_workload = self.cpu_day_hour_inflx[self.step_id] + executed_flx_workload

        # next step
        self.step_id += 1
        state = np.asarray(np.hstack(([self.cpu_day_hour_inflx, 
                            self.cpu_day_hour_flx,
                            self.queue
                            ])), dtype=np.float32)
        
        if self.step_id < self.time_steps:
            penalty = 0
            done = False
            truncated = False
        else:
            penalty = self.queue if self.queue > 0.001 else 0
            done = True
            truncated = False

        reward = - 10*penalty
        # + net_workload* self.carbon_data_day_hour[self.day_id][self.step_id]

        info = {"cpu_inflx": self.cpu_day_hour_inflx[self.step_id-1], #当前inflexible 负荷
                "cpu_flx": self.cpu_day_hour_flx[self.step_id-1],  #当前的flexible负荷
                "max_flx_execution": max_flx_execution,  #当前flexible负荷最大执行量
                "cpu_flx_executed": executed_flx_workload,  #当前执行的flexible负荷
                "cpu_load": net_workload,  #当前的flexible负荷
                "cpu_action": action,  #当前智能体的决策
                "cpu_queue": self.queue, #当前queue负荷
                "cpu_penalty": penalty}  #超时惩罚

        return state, reward, done, truncated, info
    
if __name__ == '__main__':
    # 创建环境实例
    env = CarbonLoadEnv()

    for i in range(100):
        
        # 重置环境
        day_id = np.random.randint(0,365)
        state, info = env.reset(day_id=day_id)
        # print(f"Initial state: {state}")
        # print(f"Initial info: {info}")
        
        # 运行一个完整的episode
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            # 随机动作（在[-1,1]范围内）
            # action = np.random.uniform(-1, 1)
            action = 1
            # 执行一步
            next_state, reward, done, truncated, info = env.step(action)
            
            # 打印每一步的信息
            # print(f"\nStep {step}:")
            # print(f"max_flx_execution: {info['max_flx_execution']:.6f}")
            # print(f"Action: {action:.6f}")
            # print(f"Reward: {reward:.6f}")
            # print(f"Next state: {next_state}")
            # print(f"Info: {info}")
            
            total_reward += reward
            step += 1
        
        #print(f"\nEpisode finished after {step} steps")
        print(f"Total reward: {total_reward:.10f}")

    