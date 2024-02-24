from itertools import pairwise


class Task:
    """
    描述某一任务的每一层工作量, 参数为一个工作量列表与任务下标，依次表示从前向后每一层神经网络的工作量
    split 用于继续分割这一任务块
    """

    def __init__(self, layers: list[float], index: int = None, arrive: float = None):
        self.layers = layers
        self.arrive = arrive
        self.idx = index
        self.total_workload = sum(layers)

    def __sizeof__(self):
        return len(self.layers)

    def split(self, limit_size):
        temp = []
        scheme = []
        for layer in self.layers:
            if sum(temp) + layer <= limit_size:
                temp.append(layer)
            else:
                scheme.append(Task(temp.copy(), self.idx, self.arrive))
                temp.clear()
        return scheme


EMPTY_TASK = Task([0], 0)
