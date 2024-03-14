import os 
import torch
import random
os.path.join("./")

from SacStrategy import MySAC, ReplayBuffer
from NetworkImplementation import Network
import numpy as np
# 测试 sac observe 功能

sac = MySAC(5, 6, 0.001, 0.001, 0.001, -1, 0.005, 0.9, 100, 1)

network = Network(7, 7, 10, 100, lambda: 1)

result = sac.observe(2, 2, 2, 4, 10, 0, network)

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

# 成功

# 设计存储与加载

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

    def interact(self, epoch):
        
        def summon_a_persudo_task():
            persudo_task = [np.random.normal(self.workload_mu, self.workload_sigma) for _ in range(self.scheme_length)]
            return persudo_task

        bias = [ 
            (-1, 0),
            (0, -1),
            (0, 0),
            (0, 1),
            (1, 0)
        ]

        for _ in range(epoch):
            persudo = summon_a_persudo_task()
            center_x = random.choice(range(self.network.width))
            center_y = random.choice(range(self.network.height))
            
            for step in range(self.scheme_length):
                # 当前状态 
                current_state = self.sac.obtain_state(2, center_x, center_y, self.network) 
                print(current_state)
                action = self.sac.take_action(torch.tensor(current_state).unsqueeze(dim=0))
                current, action, reward, nxt, isDone = self.sac.observe(2, center_x, center_y, action.item(), persudo[step], 1 if step == self.scheme_length - 1 else 0, self.network)
                self.buffer.add(current, action, reward, nxt, isDone)
                
                dx, dy = bias[action]
                center_x = (center_x + dx) % self.network.width 
                center_y = (center_y + dy) % self.network.height 

                # 样本数量足够了，可以训练了
                if self.buffer.size() >= self.learn_thre:
                    batch = self.buffer.sample(self.sample_size)
                    self.sac.update(batch)


# 编写模型训练代码

env = Environment(network, buffer, sac, 5, 15, 2, 10, 10)

env.interact(20)

