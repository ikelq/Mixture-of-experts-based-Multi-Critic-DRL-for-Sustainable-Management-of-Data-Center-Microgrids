import numpy as np
import matplotlib.pyplot as plt
from envs_lab.workload_env import CarbonLoadEnv
import pandas as pd

# 创建环境实例
env = CarbonLoadEnv()

# 重置环境
state, info = env.reset(day_id=0)
print(f"Initial state: {state}")
print(f"Initial info: {info}")

# 运行一个完整的episode
done = False
total_reward = 0
step = 0
info_list = []

while not done:
    # 随机动作（在[-1,1]范围内）
    # action = np.random.uniform(-1, 1)
    action = 0.5
    
    # 执行一步
    next_state, reward, done, info = env.step(action)
    
    # 打印每一步的信息
    print(f"\nStep {step}:")
    print(f"Action: {action:.3f}")
    print(f"Reward: {reward:.3f}")
    print(f"Next state: {next_state}")
    print(f"Info: {info}")
    
    # 存储info信息
    info_list.append(info)
    
    total_reward += reward
    step += 1

print(f"\nEpisode finished after {step} steps")
print(f"Total reward: {total_reward:.3f}")

# 绘制每一步info的曲线
plt.figure(figsize=(10, 6))
for key in info_list[0].keys():
    values = [info[key] for info in info_list]
    plt.plot(values, label=key)
plt.xlabel('Step')
plt.ylabel('Value')
plt.title('Info Values Over Steps')
plt.legend()
plt.grid(True)
plt.show()

# 将 info_list 转换为 DataFrame
df = pd.DataFrame(info_list)
# 保存为 CSV 文件
df.to_csv('info_data_workload.csv', index=False)
print("Info data saved to info_data_workload.csv") 