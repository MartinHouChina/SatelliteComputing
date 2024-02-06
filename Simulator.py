from NetworkImplementation import Network
from TaskImplementation import Task


def get_sliced(task: Task, segment_num: int):
    layer_cnt = len(task.layers)


class Simulator:
    def __init__(self, network: Network, task_matrix: list[list[list[Task]]]):
        self.network = network
        self.task_matrix = task_matrix
