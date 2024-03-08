import os 
import torch

os.path.join("./")

from SacStrategy import MySAC, ReplayBuffer
from NetworkImplementation import Network

# 测试 sac observe 功能
sac = MySAC(5, 1, 0.001, 0.001, 0.001, -1, 0.005, 0.9, 100, 1)

network = Network(7, 7, 10, 100, lambda: 1)

result = sac.observe(2, 2, 2, 4, 10, 0, network)

print(result, len(result))

# 成功


# 测试 ReplayBuffer

buffer = ReplayBuffer(100)

buffer.add(*result)

result = buffer.sample(1)

print(result)

# 成功


# 测试 sac update 核心函数 

sac.update(result)

