from StrategyImplementation import Strategy
from NetworkImplementation import Network
from TaskImplementation import Task
import numpy as np
import random


class GA(Strategy):
    def __init__(self, network: Network, cof_td, cof_it, cof_tt, epsilon, init_indi_num: int, max_indi_num: int,
                 max_iteration: int, inserting_num: int):
        """
        network:网络环境
        epsilon:迭代精度误差
        init_indi_num: 初始个体个数
        max_indi_num: 种群最大尺寸
        task_seq: 已切块的任务序列
        """
        self.inserting_num = inserting_num
        self.network = network
        self.cof_td = cof_td
        self.cof_tt = cof_tt
        self.cof_it = cof_it
        self.epsilon = epsilon
        self.init_indi_num = init_indi_num
        self.max_indi_num = max_indi_num
        self.max_iteration = max_iteration

    def getDeficit(self, chro: list[tuple], task_block_seq: list[list[Task]], start_x: int, start_y: int):
        total_drop, inference_time, transmission_time = 0, 0, 0
        abandoned_set = set()
        log = []
        for i in range(len(chro)):
            # 拿出这一列任务切片
            slice = [origin[i] for origin in task_block_seq if origin[i].idx not in abandoned_set]
            total_workload = np.sum([task_slice.total_workload for task_slice in slice])

            # 计算推演时间与总丢包
            now_x, now_y = chro[i]
            extra_inf_time, extra_processed_list, extra_drop_list = self.network.assign_with(now_x, now_y, slice)
            total_drop += len(extra_drop_list)
            for idx in extra_drop_list:
                abandoned_set.add(idx)
            inference_time += extra_inf_time

            # 计算运输时间
            transmission_time += total_workload * self.network.transition_cof_between(start_x, start_y, now_x, now_y)

            # 为下一次推演做准备
            start_x, start_y = now_x, now_y

            # 为还原做准备
            log.append((now_x, now_y, extra_processed_list))

        # 还原计算之前的场景
        for x, y, pro_list in log:
            for idx in pro_list:
                self.network.undo_with(x, y, idx)

        return self.cof_it * inference_time + self.cof_td * total_drop + self.cof_tt * transmission_time

    def mate(self, chro1: list[tuple], chro2: list[tuple]):
        if len(chro1) != len(chro2):
            raise RuntimeError("The length of input chromosome is not equivalent")
        Len = len(chro1)
        newborn = []
        for i in range(Len):
            for j in range(Len):
                tup1, tup2 = chro1[i], chro2[j]
                if tup1 == tup2:
                    template = chro1[:i] + chro2[j:]
                    newborn.append(template[:Len])
                    newborn.append(template[-Len:])
        return newborn

    def isValidChro(self, chro: list[tuple]):
        S = set(chro[0])
        for i in range(1, len(chro)):
            if chro[i] != chro[i - 1] and chro[i] in S:
                return False
            else:
                S.add(chro[i])
        return True

    def decide_a_scheme_for(self, mcd: int, center_x: int, center_y: int,
                            task_block: list[list[Task]], scheme_length: int) -> list[tuple]:
        # 预先运算出动作空间
        def MyRange(L, R):
            return range(L, R + 1)

        bias = [(0, i) for i in MyRange(-mcd, mcd)]

        for i in MyRange(1, mcd):
            bias.extend([(i, 0), (-i, 0)])
            for j in MyRange(1, mcd - i):
                bias.extend([(i, j), (i, -j), (-i, j), (-i, -j)])

        bias = list(map(lambda x: (x[0] + center_x, (x[1] + center_y) % self.network.height), bias))
        action_space = list(filter(lambda x: 0 <= x[0] < self.network.width, bias))
        # 初始化群体

        group = []
        while len(group) < self.init_indi_num:
            individual = random.choices(action_space, k=scheme_length)
            if self.isValidChro(individual):
                group.append((self.getDeficit(individual, task_block, center_x, center_y), individual))

        # 迭代
        last_best_deficit = 1 ** 20
        for _ in range(self.max_iteration):
            # 发现迭代缓慢则直接退出迭代
            if last_best_deficit - group[0][0] < self.epsilon:
                break

            # 繁殖
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    newborn_group = self.mate(group[i][1], group[j][1])
                    newborn_group = list(filter(lambda x: self.isValidChro(x), newborn_group))
                    for newchro in newborn_group:
                        group.append((self.getDeficit(newchro, task_block, center_x, center_y), newchro))

            # 插入
            for __ in range(self.inserting_num):
                individual = random.choices(action_space, k=scheme_length)
                if self.isValidChro(individual):
                    group.append((self.getDeficit(individual, task_block, center_x, center_y), individual))

            # 淘汰
            group.sort(lambda x: x[0])
            group = group[:self.max_indi_num]

        # 返回最优个体染色体
        return group[0][1]


class Random(Strategy):
    def __init__(self, network: Network):
        self.network = network

    def decide_a_scheme_for(self, mcd: int, center_x: int, center_y: int,
                            task_block: list[Task], scheme_length: int) -> list[tuple]:

        # 预先运算出动作空间
        def MyRange(L, R):
            return range(L, R + 1)

        bias = [(0, i) for i in MyRange(-mcd, mcd)]

        for i in MyRange(1, mcd):
            bias.extend([(i, 0), (-i, 0)])
            for j in MyRange(1, mcd - i):
                bias.extend([(i, j), (i, -j), (-i, j), (-i, -j)])

        bias = list(map(lambda x: (x[0] + center_x, (x[1] + center_y) % self.network.height), bias))
        action_space = list(filter(lambda x: 0 <= x[0] < self.network.width, bias))

        scheme = random.choices(action_space, k=scheme_length)
        return scheme


class Greedy(Strategy):
    def __init__(self, network: Network):
        self.network = network

    def decide_a_scheme_for(self, mcd: int, center_x: int, center_y: int,
                            task_block: list[Task], scheme_length: int) -> list[tuple]:
        def MyRange(L, R):
            return range(L, R + 1)

        bias = [(0, i) for i in MyRange(-mcd, mcd)]

        for i in MyRange(1, mcd):
            bias.extend([(i, 0), (-i, 0)])
            for j in MyRange(1, mcd - i):
                bias.extend([(i, j), (i, -j), (-i, j), (-i, -j)])

        bias = list(map(lambda x: (x[0] + center_x, (x[1] + center_y) % self.network.height), bias))
        action_space = list(filter(lambda x: 0 <= x[0] < self.network.width, bias))

        scheme = random.choices(action_space, k=scheme_length)
        return scheme
