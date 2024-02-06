from itertools import pairwise


class Task:
    """
    描述某一任务的每一层工作量, 参数为一个工作量列表与任务下标，依次表示从前向后每一层神经网络的工作量
    split 用于继续分割这一任务块
    """

    def __init__(self, layers: list[float], index: int):
        self.layers = layers
        self.index = index
        self.total_workload = sum(layers)

    def __sizeof__(self):
        return len(self.layers)

    def split(self, slicing_point: list[int]):

        for point in slicing_point:
            if point < 0 or point >= self.__sizeof__():
                raise RuntimeError("The slicing point is out of bound!")

        slicing_point.sort()

        for pre, suf in pairwise(slicing_point):
            if pre == suf:
                raise RuntimeError("There should not be 2 same slicing point!")

        slicing_result = []
        endpoint = 0

        for point in slicing_point:
            slice = Task(self.layers[endpoint:point], self.index)
            slicing_result.append(slice)

        return slicing_result


if __name__ == '__main__':
    task = Task([1, 2, 4, 5], 2)
    task_slices = task.split([1, 2])
    print(task, *task_slices)
