from NetworkImplementation import Network
from TaskImplementation import Task
from TaskImplementation import EMPTY_TASK
from EventSimulator import Event
from StrategyImplementation import Strategy
from queue import PriorityQueue
import random


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
    template_empty_task = EMPTY_TASK
    template_empty_task.arrive = task.arrive
    while len(scheme) < num_segments: scheme.append(template_empty_task)

    return scheme


class Simulator:
    def __init__(self, network: Network, task_matrix: list[list[list[Task]]], segment_num: int):
        self.network = network
        self.task_matrix = task_matrix
        self.segment_num = segment_num

    def preprocess(self, random_function, max_allowed_workload_for_blocks: float, *args):
        """
        为任务切块、分配下标，以及生成抵达时间
        random_function 是函数,可以使用 TaskSummoner 里的函数
        args 是参数
        segment_num 是期望切割段数
        """

        def divide_into_blocks(task_list: list[Task], max_allow_workload: float) -> list[list[Task]]:
            if not len(task_list):
                return []
            res, block = [], []
            max_allow_workload = 1 + max(max([task.total_workload for task in task_list]), max_allow_workload)
            cur_workload = 0
            for task in task_list:
                if cur_workload + task.total_workload <= max_allow_workload:
                    block.append(task)
                    cur_workload += task.total_workload
                else:
                    res.append(block)
                    block = [task]
                    cur_workload = task.total_workload
            return res

        # 给每个卫星的任务列表打乱后分块
        for i in range(len(self.task_matrix)):
            for j in range(len(self.task_matrix[i])):
                random.shuffle(self.task_matrix[i][j])
                self.task_matrix[i][j] = divide_into_blocks(self.task_matrix[i][j], max_allowed_workload_for_blocks)

        indices = 0
        # 行
        for i in range(len(self.task_matrix)):
            # 纵
            for j in range(len(self.task_matrix[i])):
                # 任务块
                for k in range(len(self.task_matrix[i][j])):
                    # 任务
                    for l in range(len(self.task_matrix[i][j][k])):
                        self.task_matrix[i][j][k][l].idx = indices
                        self.task_matrix[i][j][k][l].arrive = random_function(*args)[0]
                        self.task_matrix[i][j][k][l] = decide_partitioning(self.task_matrix[i][j][k][l],
                                                                           self.segment_num)
                        indices += 1

    def simulate_with(self, mcd: int, strategy: Strategy, transmission_latency):
        # 模拟时间线
        eventline = PriorityQueue()

        # 加入所有任务
        for i in range(len(self.task_matrix)):
            for j in range(len(self.task_matrix[i])):
                satellite = self.task_matrix[i][j]
                for task_block in satellite:
                    arrive = task_block[0][0].arrive
                    eventline.put(Event(i, j, task_block, 0, arrive, self.network))

        total_drop, total_processed, total_delay = 0, 0, 0

        # 开始模拟
        while eventline.qsize():
            event: Event = eventline.get()
            print(event.dfn, event.status, total_drop, total_processed)
            if event.status == "undetermined":
                event.determine_with_strategy(strategy, mcd, self.segment_num)
                eventline.put(event)
            elif event.status != "finished":
                event.execute()
                eventline.put(event)
            else:
                total_drop += event.total_drop
                total_processed += event.total_successful
                total_delay += event.total_time



if __name__ == '__main__':
    from StrategyImplementation import GA
    from TaskSummoner import *

    network = Network(7, 7, 2, 100, lambda: 15)
    ga_decider = GA(network, 10 ** 6, 1, 1, 10, 10, 15, 20, 5)

    task_matrix = summon_task_matrix(7, 7, 7, 4, 10, 20)
    simulator = Simulator(network, task_matrix, 3)
    simulator.preprocess(uniform_distribution, 300, 10, 200, 1)
    simulator.simulate_with(2, ga_decider, transmission_latency=20)
