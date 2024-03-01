# 处理离散问题的模型
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import collections
import random
from abc import abstractmethod


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
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    # 当前时刻的经验池容量
    def size(self):
        return len(self.buffer)


# ----------------------------------------- #
# 策略网络
# ----------------------------------------- #

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, min_log_std=-10, max_log_std=2):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, action_dim)  # 均值
        self.log_std_head = nn.Linear(256, action_dim)  # 标准差
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std_head = F.relu(self.log_std_head(x))
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
        return mu, log_std_head


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
        s = s.reshape(-1)
        a = a.reshape(-1)
        x = torch.cat((s, a), -1)  # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ----------------------------------------- #
# SAC 强化学习系统 构建
# ----------------------------------------- #

class SAC:
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
        self.q_net_1 = QNet(n_states + n_actions).to(device)
        self.q_net_2 = QNet(n_states + n_actions).to(device)

        # 实例化两个 Q-net 目标网络
        self.target_q_net_1 = QNet(n_states + n_actions).to(device)
        self.target_q_net_2 = QNet(n_states + n_actions).to(device)

        # 预测和目标的价值网络的参数初始化一样
        self.target_q_net_1.load_state_dict(self.q_net_1.state_dict())
        self.target_q_net_2.load_state_dict(self.q_net_2.state_dict())

        # 确定网络所使用的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q_net_1_optimizer = torch.optim.Adam(self.q_net_1.parameters(), lr=Q_lr)
        self.q_net_2_optimizer = torch.optim.Adam(self.q_net_2.parameters(), lr=Q_lr)

        # 初始化可训练参数alpha
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        # alpha可以训练求梯度
        self.log_alpha.requires_grad = True
        # 定义alpha的优化器
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # 属性分配
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.rho = rho
        self.device = device

    def take_action(self, state):  # 对应算法第 4 行, 利用 reparameterization trick 选取动作
        state = torch.FloatTensor(state).to(self.device)
        mu, log_sigma = self.actor(state)
        sigma = torch.exp(log_sigma)
        dist = torch.distributions.Normal(mu, sigma)
        z = dist.sample()
        action = torch.tanh(z).detach().cpu().numpy()
        return action.item()

    @abstractmethod
    def observe(self, current_state, action, environment):  # 对应算法 5 ~ 6 行
        # 在特定环境中的指定状态下执行某一动作后, 观察得到的奖励，下一状态，以及终止标志
        pass

    def calc_target(self, rewards, next_states, dones):  # 对应算法第 12 行, 计算目标值 y

        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        # 根据当前策略网络中提取下一个可能动作
        next_action = self.take_action(next_states)

        # 目标价值网络，下一时刻的 state_value  [b,n_actions]
        q1_value = self.target_q_net_1(next_states, next_action)
        q2_value = self.target_q_net_2(next_states, next_action)

        # 算出下一个价值
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy

        # 时序差分，目标网络输出当前时刻的state_value  [b, n_actions]
        y = rewards + self.gamma * next_value * (1 - dones)
        return y

    # 模型训练
    def update(self, batch):  # 对应 11 ~ 15 行
        states = torch.tensor(batch['states'], dtype=torch.float).to(self.device)  # [b,n_states]
        actions = torch.tensor(batch['actions']).view(-1, 1).to(self.device)  # [b,1]
        rewards = torch.tensor(batch['rewards'], dtype=torch.float).view(-1, 1).to(self.device)  # [b,1]
        next_states = torch.tensor(batch['next_states'], dtype=torch.float).to(self.device)  # [b,n_states]
        dones = torch.tensor(batch['dones'], dtype=torch.float).view(-1, 1).to(self.device)  # [b,1]

        # --------------------------------- #
        # 更新2个价值网络, 对应 12 ~ 13 行
        # --------------------------------- #

        # 目标网络的state_value [b, 1]
        y = self.calc_target(rewards, next_states, dones)

        # Q网络1--预测
        q_net_1_qvalues = self.q_net_1(states).gather(1, actions)
        # 均方差损失 预测-目标
        q_net_1_loss = torch.mean(F.mse_loss(q_net_1_qvalues, y.detach()))

        # Q网络2--预测
        q_net_2_qvalues = self.q_net_2(states).gather(1, actions)
        # 均方差损失
        q_net_2_loss = torch.mean(F.mse_loss(q_net_2_qvalues, y.detach()))

        # 梯度清0
        self.q_net_1_optimizer.zero_grad()
        self.q_net_2_optimizer.zero_grad()

        # 梯度反向传播
        q_net_1_loss.backward()
        q_net_2_loss.backward()

        # 梯度更新
        self.q_net_1_optimizer.step()
        self.q_net_2_optimizer.step()

        # --------------------------------- #
        # 更新策略网络, 对应第 14 行
        # --------------------------------- #

        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.q_net_1(states, new_actions)
        q2_value = self.q_net_2(states, new_actions)
        actor_loss = torch.mean(- self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --------------------------------- #
        # 更新可训练遍历alpha
        # --------------------------------- #

        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        # 梯度更新
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # 软更新目标价值网络 第 15 行
        self.soft_update(self.q_net_1, self.target_q_net_1)
        self.soft_update(self.q_net_2, self.target_q_net_2)

    # 软更新，每次训练更新部分参数
    def soft_update(self, net, target_net):  # 对应算法第 15 行
        # 遍历预测网络和目标网络的参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # 预测网络的参数赋给目标网络
            param_target.data.copy_(param_target.data * (1 - self.rho) + param.data * self.rho)
