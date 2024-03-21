from functools import total_ordering
import os 
import torch
import random
os.path.join("./")

from SacStrategy import MySAC, ReplayBuffer
from NetworkImplementation import Network
import numpy as np
# 测试 sac observe 功能

sac = MySAC(14, 13, 0.01,0.01, 0.01, -1, 0.5, 0.9, 0.1, 10)

network = Network(7, 7, 10, 100, lambda: 1)

result = sac.observe(2, 2, 2, 4, 10, 10, 0, network)
print(result)
# 成功


# 测试 ReplayBuffer

buffer = ReplayBuffer(100)
buffer.add(*result)
buffer.add(*result)
buffer.add(*result)
buffer.add(*result)

result = buffer.sample(3)


# 成功


# 测试 sac update 核心函数 
sac.update(result)
print("YES")
sac.update(result)
print("YEs")
sac.update(result)


sac.save_param("sac_model/")

sac.load_param("sac_model/")

# 成功

# 编写环境类

class Environment:
    def __init__(
            self,
            network: Network,
            buffer: ReplayBuffer,
            sac: MySAC, # 任务块切片个数
            scheme_length: int,
            workload_mu, # 任务块切片工作均值
            workload_sigma, # 任务快切片标准差
            learn_thre: int, # 开始学习的门槛
            sample_size: int # 抽样大小
        ):
        self.network = network
        self.buffer = buffer
        self.sac = sac
        self.scheme_length = scheme_length
        self.workload_mu = workload_mu
        self.workload_sigma = workload_sigma
        self.learn_thre = learn_thre
        self.sample_size = sample_size

    def interact(self, epoch, scale, selected_x=None, selected_y=None, persudo=None):
    
        def summon_a_persudo_task():
            persudo_task = [np.random.normal(self.workload_mu, self.workload_sigma) for _ in range(self.scheme_length)]
            return persudo_task
        
        result = []
    
        for round in range(epoch):
            total_reward = 0
            for _ in range(scale):
                persudo = summon_a_persudo_task() if persudo is None else persudo 
                center_x = random.choice(range(self.network.width)) if selected_x is None else selected_x
                center_y = random.choice(range(self.network.height)) if selected_y is None else selected_y
                
                bias = self.network.calc_action_space(2, center_x, center_y, True)

                for step in range(self.scheme_length):
                    # 当前状态 
                    current_state = self.sac.obtain_state(2, center_x, center_y, persudo[step], self.network)

                    # 更新 buffer
                    action = self.sac.take_action(torch.tensor(current_state).unsqueeze(dim=0))
                    current, action, reward, nxt, isDone = self.sac.observe(2, center_x, center_y, action.item(), persudo[step],0 if step == self.scheme_length - 1 else persudo[step + 1], 1 if step == self.scheme_length - 1 else 0, self.network)
                    self.buffer.add(current, action, reward, nxt, isDone)
                    total_reward += reward

                    # 更新执行中心
                    dx, dy = bias[action]
                    center_x = (center_x + dx) % self.network.width 
                    center_y = (center_y + dy) % self.network.height 

                    # 更新 network 
                    workload = persudo[step]
                    network.satellite_table[center_x][center_y].capability -= workload
                    

                     # 样本数量足够了，可以训练了
                    if self.buffer.size() >= self.learn_thre:
                        batch = self.buffer.sample(self.sample_size)
                        self.sac.update(batch)
           
            network.reset()
            self.buffer.reset()

# 编写模型训练代码

env = Environment(network, buffer, sac, 5, 15, 2, 10, 5)
print(list(sac.target_q_net_1.parameters()))
ans = env.interact(100, 20, 3, 3, [20, 20, 20, 20, 20])

print(list(sac.target_q_net_1.parameters()))
print(ans)

