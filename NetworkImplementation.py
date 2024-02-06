from SatelliteImplementation import Satellite
import numpy as np


class Network:
    def __init__(self, width: int, height: int, capability: int, mcd: int):
        """
        构造卫星网络
        :param width: 宽
        :param height: 长
        :param capability: 每颗卫星的计算资源
        :param mcd: 最大可交流距离
        """
        self.width = width
        self.height = height
        self.satellite_table = [[Satellite(capability) for _ in range(height)] for __ in range(width)]
        self.mcd = mcd

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

    def __getitem__(self, item):
        return self.satellite_table[item]
