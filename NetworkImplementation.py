from SatelliteImplementation import Satellite
import numpy as np
from TaskImplementation import Task
from typing import Callable


class Network:
    def __init__(self, width: int, height: int, bandwidth: float, capability: int, transition_distribution: Callable):
        """
        构造卫星网络
        :param width: 宽
        :param height: 长
        :param capability: 每颗卫星的计算资源
        :param mcd: 最大可交流距离
        :param max_variance: 出现过的最大资源分布方差
        """
        self.max_variance = 0
        self.width = width
        self.height = height
        self.transition_distribution = transition_distribution
        self.satellite_table = [
            [Satellite(capability) for _ in range(height)] for __ in range(width)]
        self.bandwidth = bandwidth

    def calc_resource_variance(self):
        """
        返回资源剩余的方差
        """
        resource_distribution = [
            [satellite.capability for satellite in row] for row in self.satellite_table]
        variance = np.var(resource_distribution)
        return variance

    def print_resource_matrix(self):
        """
        打印资源剩余矩阵
        """
        for i in range(self.width):
            print([self.satellite_table[i][j].capability for j in range(self.height)])

    def calc_task_completion(self):
        """
        返回当前总完成率
        """
        completed = np.sum([np.sum([satellite.completed_cnt for satellite in row])
                           for row in self.satellite_table])
        uncompleted = np.sum([np.sum(
            [satellite.abandon_cnt for satellite in row]) for row in self.satellite_table])
        return completed / (completed + uncompleted)

    def transition_cof_between(self, cord_x1, cord_y1, cord_x2, cord_y2):
        distance = min([abs(cord_x1 - cord_x2), self.width - 1 - cord_x2 + cord_x1, self.width - 1 - cord_x1 + cord_x2]) + \
            min([abs(cord_y1 - cord_y2), self.height - 1 - cord_y2 + cord_y1, self.height - 1 - cord_y1 + cord_y2])
        return distance * self.transition_distribution() / self.bandwidth

    def __getitem__(self, item):
        return self.satellite_table[item]

    def assign_with(self, cord_x, cord_y, task_block: list[Task]) -> tuple:
        """
        为第 x 行 第 y 列的卫星分配 任务task, 返回处理列表，详见 satellite.load 函数
        """
        if cord_x < 0 or cord_x >= self.width or cord_y < 0 or cord_y >= self.height:
            raise RuntimeError("An invalid coordinate!!!")
        result = self.satellite_table[cord_x][cord_y].load(task_block)
        self.max_variance = max(self.max_variance,self.calc_resource_variance())
        return result


    def undo_with(self, cord_x, cord_y, task_idx):
        if cord_x < 0 or cord_x >= self.width or cord_y < 0 or cord_y >= self.height:
            raise RuntimeError("An invalid coordinate!!!")
        self.satellite_table[cord_x][cord_y].offload(task_idx)
        self.max_variance = max(self.max_variance, self.calc_resource_variance())

    def reset(self):
        self.max_variance = 0
        for i in range(self.width):
            for j in range(self.height):
                self.satellite_table[i][j].reset()

    def calc_action_space(self, mcd: int, center_x: int, center_y: int):
        """
        计算以 center_x, center_y 为中心的 卫星 在 mcd 下的 动作空间
        """

        bias = [(0, i) for i in range(-mcd, mcd + 1)]

        for i in range(1, mcd + 1):
            bias.extend([(i, 0), (-i, 0)])
            for j in range(1, mcd - i + 1):
                bias.extend([(i, j), (i, -j), (-i, j), (-i, -j)])

        action_space = list(map(lambda x: (
            (x[0] + center_x) % self.width, (x[1] + center_y) % self.height), bias))
        return action_space
    
    def calc_finished_task(self):
        completed = np.sum([np.sum([satellite.completed_cnt for satellite in row]) for row in self.satellite_table])
        return completed

if __name__ == '__main__':
    network = Network(7, 7, 2, 100, lambda: 15)
    task = Task([11], 1, 1)
    network.assign_with(1, 1, [task])
    network.print_source_matrix()
    network.undo_with(1, 1, 1)
    network.print_source_matrix()
