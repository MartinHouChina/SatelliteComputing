from SatelliteImplementation import Satellite
import numpy as np
from TaskImplementation import Task, EMPTY_TASK
from typing import Callable


class Network:
    def __init__(self, width: int, height: int, bandwidth: float, capability: int, transition_distribution: Callable):
        """
        构造卫星网络
        :param width: 宽
        :param height: 长
        :param capability: 每颗卫星的计算资源
        :param mcd: 最大可交流距离
        """
        self.width = width
        self.height = height
        self.transition_distribution = transition_distribution
        self.satellite_table = [[Satellite(capability) for _ in range(height)] for __ in range(width)]
        self.bandwidth = bandwidth

    def calc_resource_variance(self):
        """
        返回资源剩余的方差
        """
        resource_distribution = [[satellite.capability for satellite in row] for row in self.satellite_table]
        variance = np.var(resource_distribution)
        return variance

    def calc_task_completion(self):
        """
        返回当前总完成率
        """
        completed = np.sum([np.sum([satellite.completed_cnt for satellite in row]) for row in self.satellite_table])
        uncompleted = np.sum([np.sum([satellite.abandon_cnt for satellite in row]) for row in self.satellite_table])
        return completed / (completed + uncompleted)

    def transition_cof_between(self, cord_x1, cord_y1, cord_x2, cord_y2):
        distance = abs(cord_x1 - cord_x2) + abs(cord_y1 - cord_y2)
        return distance * self.transition_distribution() / self.bandwidth

    def __getitem__(self, item):
        return self.satellite_table[item]

    def assign_with(self, cord_x, cord_y, task_block: list[Task]):
        """
        为第 x 行 第 y 列的卫星分配 任务task
        """
        if cord_x < 0 or cord_x >= self.width or cord_y < 0 or cord_y >= self.height:
            raise RuntimeError("An invalid coordinate!!!")
        return self.satellite_table[cord_x][cord_y].load(task_block)

    def undo_with(self, cord_x, cord_y, task_idx):
        if cord_x < 0 or cord_x >= self.width or cord_y < 0 or cord_y >= self.height:
            raise RuntimeError("An invalid coordinate!!!")
        self.satellite_table[cord_x][cord_y].offload(task_idx)
