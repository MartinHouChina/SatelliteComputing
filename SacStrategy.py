# 处理离散问题的模型
import torch
from torch import nn
from torch.fx.node import Target
from torch.nn import functional as F
import numpy as np
import collections
import random
from abc import abstractmethod, ABC
from NetworkImplementation import Network
import os
from torch.distributions import Categorical

# ----------------------------------------- #
# 经验回放池
# ----------------------------------------- #

class ReplayBuffer:
    def __init__(self, capacity):  # 经验池容量
        self.buffer = collections.deque(maxlen=capacity)  # 队列，先进先出

    # 经验池增加
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # 随机采样batch组
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # 取出这batch组数据
        # state, action, reward, next_state, done = zip(*transitions)
        state, action, reward, next_state, done = tuple([item[i] for item in transitions] for i in range(5))
        return state, action, reward, next_state, done

    # 当前时刻的经验池容量
    def size(self):
        return len(self.buffer)

    def reset(self):
        self.buffer.clear()


# ----------------------------------------- #
# 离散 策略网络 输入状态 输出 指定动作
# ----------------------------------------- #

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)  # 标准差

    def forward(self, s):
        probs = F.softmax(self.fc3(F.relu(self.fc2(F.relu(self.fc1(s))))), dim=-1)
        a = Categorical(probs).sample()
        a = a.unsqueeze(dim=1)
        return a, probs


# ----------------------------------------- #
# 动作 - 状态价值网络
# ----------------------------------------- #

class QNet(nn.Module):
    def __init__(self, input_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, s, a):
        x = torch.cat((s, a), 1)  # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ----------------------------------------- #
# SAC 强化学习系统核心，对于观察部分需要根据具体实验修改
# ----------------------------------------- #

class SAC_core(ABC):
    def __init__(self,
                 n_states,  # 状态数
                 n_actions,  # 动作数
                 actor_lr,  # policy 学习率
                 Q_lr,  # QNet 学习率
                 alpha_lr,  # alpha 学习率
                 target_entropy,
                 rho,  # 参数更新率
                 gamma,  # 折扣因子
                 device="gpu" if torch.cuda.is_available() else "cpu"  # 训练设备
                 ):
        # 实例化策略网络
        self.actor = PolicyNet(n_states, n_actions).to(device)

        # 实例化两个 Q-net 预测网络
        self.q_net_1 = QNet(n_states + 1).to(device)
        self.q_net_2 = QNet(n_states + 1).to(device)

        # 实例化两个 Q-net 目标网络
        self.target_q_net_1 = QNet(n_states + 1).to(device)
        self.target_q_net_2 = QNet(n_states + 1).to(device)

        # 预测和目标的价值网络的参数初始化一样
        self.target_q_net_1.load_state_dict(self.q_net_1.state_dict())
        self.target_q_net_2.load_state_dict(self.q_net_2.state_dict())

        # 确定网络所使用的优化器
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr)
        self.q_net_1_optimizer = torch.optim.Adam(
            self.q_net_1.parameters(), lr=Q_lr)
        self.q_net_2_optimizer = torch.optim.Adam(
            self.q_net_2.parameters(), lr=Q_lr)

        # 初始化可训练参数alpha
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        # alpha可以训练求梯度
        self.log_alpha.requires_grad = True
        # 定义alpha的优化器
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr)

        # 属性分配
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.rho = rho
        self.device = device

    @abstractmethod
    def obtain_state(self, *args):  # 获取当前状态
        pass

    def take_action(self, state):  # 对应算法第 4 行, 利用 reparameterization trick 选取动作
        state = state.clone().detach().to(self.device)

        action, _ = self.actor(state)
        return action

    @abstractmethod
    def observe(self, *args):  # 对应算法 5 ~ 6 行
        # 在特定环境中的指定状态下执行某一动作后, 观察得到的奖励，下一状态，以及终止标志
        pass

    def calc_target(self, rewards, next_states, dones):  # 对应算法第 12 行, 计算目标值 y

        _, probs = self.actor(next_states)
        entropy = - (probs * torch.log(probs)).sum()
        # 根据当前策略网络中提取下一个可能动作
        next_action = self.take_action(next_states)
        # 目标价值网络，下一时刻的 state_value  [b,n_actions]
        q1_value = self.target_q_net_1(next_states, next_action)
        q2_value = self.target_q_net_2(next_states, next_action)
        # 算出下一个价值
        next_value = torch.min(q1_value, q2_value) + \
                     self.log_alpha.exp() * entropy
        # 时序差分，目标网络输出当前时刻的state_value  [b, n_actions]
        y = rewards + self.gamma * next_value * (1 - dones)
        return y

    # 模型训练
    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.tensor(states, dtype=torch.float).to(self.device)  # [b,n_states]
        actions = torch.tensor(actions, dtype=torch.long).view(-1, 1).to(self.device)  # [b,1]
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)  # [b,1]
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)  # [b,n_states]
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)  # [b,1]

        y = self.calc_target(rewards, next_states, dones)

        q_net_1_qvalues = self.q_net_1(states, actions)
        q_net_1_loss = torch.nn.MSELoss()(q_net_1_qvalues, y.detach())

        q_net_2_qvalues = self.q_net_2(states, actions)
        q_net_2_loss = torch.nn.MSELoss()(q_net_2_qvalues, y.detach())

        q_net_1_loss.backward()
        q_net_2_loss.backward()
    
        self.q_net_1_optimizer.step()
        self.q_net_2_optimizer.step()
        self.q_net_1_optimizer.zero_grad()
        self.q_net_2_optimizer.zero_grad()

        new_actions, probs = self.actor(states)
        entropy = - (probs * torch.log(probs)).sum()
        q1_value = self.q_net_1(states, new_actions)
        q2_value = self.q_net_2(states, new_actions)
        actor_loss = (self.log_alpha.exp() * probs - torch.min(q1_value, q2_value)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        self.soft_update(self.q_net_1, self.target_q_net_1)
        self.soft_update(self.q_net_2, self.target_q_net_2)

    # 软更新，每次训练更新部分参数
    def soft_update(self, net, target_net):  # 对应算法第 15 行
        # 遍历预测网络和目标网络的参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # 预测网络的参数赋给目标网络
            param_target.data.copy_(
                param_target.data * (1 - self.rho) + param.data * self.rho)

    def save_param(self, path):
        # 确保路径存在
        assert os.path.exists(path)

        torch.save(self.q_net_1, path + 'q_net_1.pth')
        torch.save(self.q_net_2, path + 'q_net_2.pth')
        torch.save(self.target_q_net_1, path + 'target_q_net_1.pth')
        torch.save(self.target_q_net_2, path + 'target_q_net_2.pth')

    def load_param(self, directory):
        # 确保该路径下有 q_net, target_q_net 相关的 4 个模型
        assert os.path.exists(directory)

        self.q_net_1 = torch.load(directory + 'q_net_1.pth')
        self.q_net_2 = torch.load(directory + 'q_net_2.pth')
        self.target_q_net_1 = torch.load(directory + 'target_q_net_1.pth')
        self.target_q_net_2 = torch.load(directory + 'target_q_net_2.pth')


class MySAC(SAC_core):  # 根据我们实验修改的 SAC 实现
    def __init__(self,
                 n_states,  # 状态数
                 n_actions,  # 动作数
                 actor_lr,  # policy 学习率
                 Q_lr,  # QNet 学习率
                 alpha_lr,  # alpha 学习率
                 target_entropy,
                 rho,  # 参数更新率
                 gamma,  # 折扣因子
                 assignment_reward,  # 成功分配后的 Reward
                 variance_ratio_reward,  # 分配前后方差的 Reward
                 device="cuda" if torch.cuda.is_available() else "cpu"  # 训练设备
                 ):
        super().__init__(n_states, n_actions, actor_lr, Q_lr, alpha_lr, target_entropy, rho,
                         gamma, device)
        self.assignment_reward = assignment_reward
        self.variance_ratio_reward = variance_ratio_reward

    def obtain_state(self, mcd, cord_x, cord_y, workload, network: Network):

        bias = network.calc_action_space(mcd, cord_x, cord_y, True)
        bias.sort()
        state = list(map(lambda c: network.satellite_table[c[0]][c[1]].capability / network.max_capability, bias))
        state = state + [workload / network.max_capability]
        return state

    def observe(self, mcd, cord_x, cord_y, next_hop, workload, nxt_workload, isDone, network: Network):  # 对应算法 5 ~ 6 行
        # 在特定环境中的指定状态下执行某一动作后, 观察得到的奖励，下一状态，以及终止标志
        # 根据我们的算法 current_state 会是一个 5 个元素 的邻近资源 tensor
        # 计算 SAC 
        bias = network.calc_action_space(mcd, cord_x, cord_y, True)
        bias.sort()
        cur_state = list(map(lambda c: network.satellite_table[(cord_x + c[0] + network.width) % network.width][(c[1] + cord_y + network.height) % network.height].capability / network.max_capability,
                             bias))
        cur_state = cur_state + [workload / network.max_capability]
        action = torch.tensor(next_hop)
        next_hop = bias[next_hop]

        reward = 0
        tar_x = (network.width + cord_x + next_hop[0]) % network.width
        tar_y = (network.height + cord_y + next_hop[1]) % network.height
        remaining = network.satellite_table[tar_x][tar_y].capability

        if remaining >= workload:

            reward += self.assignment_reward

            network.satellite_table[tar_x][tar_y].capability -= workload
            bias = network.calc_action_space(mcd, cord_x, cord_y, True)
            bias.sort()
            nxt_state = list(map(lambda c: network.satellite_table[(cord_x + c[0] + network.width) % network.width][(c[
                                                                                                                         1] + cord_y + network.height) % network.height].capability / network.max_capability,
                                 bias))
            nxt_state = nxt_state + [nxt_workload / network.max_capability]
            network.satellite_table[tar_x][tar_y].capability += workload
        else:
            nxt_state = cur_state

        reward += self.variance_ratio_reward * max(0, np.var(cur_state) - np.var(nxt_state))

        return cur_state, action, reward, nxt_state, isDone
