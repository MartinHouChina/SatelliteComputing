from NetworkImplementation import Network
from TaskImplementation import Task
from TaskImplementation import EMPTY_TASK
from typing import Callable
from StrategyImplementation import Strategy


def decide_partitioning(task: Task, num_segments: int) -> list[Task]:
    L, R = max(task.layers) + 1, sum(task.layers) * 2
    epsilon = 1e-3
    while R - L > epsilon:
        mid = (L + R) / 2
        if len(task.split(mid)) > num_segments:
            L = mid
        else:
            R = mid
    scheme = task.split(L)

    while len(scheme) < num_segments: scheme.append(EMPTY_TASK)

    return scheme


class Event:
    def __init__(self, builtin_function: Callable, dfn: float, *args):
        self.function = builtin_function
        self.args = args
        self.dfn = float

    def __lt__(self, other):
        return self.dfn < other.dfn

    def execute(self):
        self.function(self.args)


class Simulator:
    def __init__(self, network: Network, task_matrix: list[list[list[Task]]]):
        self.network = network
        self.task_matrix = task_matrix

    def preprocess(self, segment_num: int, random_function, *args):
        """
        为任务切块、分配下标，以及生成抵达时间
        random_function 是函数,可以使用 TaskSummoner 里的函数
        args 是参数
        segment_num 是期望切割段数
        """
        indices = 0
        for i in range(len(self.task_matrix)):
            for j in range(len(self.task_matrix[i])):
                for k in range(len(self.task_matrix[i][j])):
                    self.task_matrix[i][j][k].idx = indices
                    self.task_matrix[i][j][k].arrive = random_function(*args)
                    self.task_matrix[i][j][k] = decide_partitioning(self.task_matrix[i][j][k], segment_num)
                    indices += 1

    def simulate_with(self, mcd: int, strategy: Strategy, transmission_latency):
        strategy.decide_a_scheme_for(mcd, )


if __name__ == '__main__':
    from StrategyImplementation import GA
    from TaskSummoner import *

    network = Network(7, 7, 2, 100, custom_distribution(15))
    ga_decider = GA(network, 10 ** 6, 1, 1, 10, 10, 15, 20, 5)

    task_matrix = summon_task_matrix(7, 7, 7, 4, 10, 20)
    simulator = Simulator(network, task_matrix)
    simulator.preprocess(3, uniform_distribution, 10, 200, 1)
    Simulator.simulate_with(2, ga_decider, transmission_latency=20)
