from NetworkImplementation import Network
from abc import abstractmethod, ABC
from TaskImplementation import Task
from typing import Callable
from SatelliteImplementation import Satellite


class Strategy(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def decide_a_scheme_for(self, determined_satellite: Satellite, task_block: list[Task], scheme_length: int):
        pass


class GA(Strategy):
    def __init__(self, network: Network, epsilon, init_indi_num,
                 max_indi_num, max_iteration):
        """
        network:网络环境
        epsilon:迭代精度误差
        init_indi_num: 初始个体个数
        max_indi_num: 种群最大尺寸
        task_seq: 已切块的任务序列
        """
        self.network = network
        self.epsilon = epsilon
        self.init_indi_num = init_indi_num
        self.max_indi_num = max_indi_num
        self.max_iteration = max_iteration

    def mate(self, chro1: list[tuple], chro2: list[tuple]):
        pass

    def decide_a_scheme_for(self, determined_satellite: Satellite, task_block: list[Task], scheme_length: int):
        pass
