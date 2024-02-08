from NetworkImplementation import Network
from TaskImplementation import Task
from TaskImplementation import EMPTY_TASK
from typing import Callable


def decide_partitioning(task: Task, num_segments: int) -> list[Task]:
    L, R = max(task.layers), sum(task.layers)
    epsilon = 1e-3
    while R - L > epsilon:
        mid = (L + R) / 2
        if len(task.split(mid)) < num_segments:
            L = mid
        else:
            R = mid
    scheme = task.split(L)
    while len(scheme) < L: scheme.append(EMPTY_TASK)
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
                    self.task_matrix[i][j][k].index = indices
                    self.task_matrix[i][j][k].arrive = random_function(args)
                    self.task_matrix[i][j][k] = decide_partitioning(self.task_matrix[i][j][k], segment_num)
                    indices += 1

    def simulate_with(self, mcd: int, strategy: Strategy, transmission_latency):


if __name__ == '__main__':
    task = Task([1, 2, 4, 5], None)
    task = decide_partitioning(task, 4)
    print(task[0].index)
