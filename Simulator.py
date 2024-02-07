from NetworkImplementation import Network
from TaskImplementation import Task
from TaskImplementation import EMPTY_TASK
import numpy as np


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


class Simulator:
    def __init__(self, network: Network, task_matrix: list[list[list[Task]]], segment_num: int):
        self.network = network
        self.task_matrix = task_matrix
        self.segment_num = segment_num

    def assign_indices(self):
        indices = 0
        for i in range(len(self.task_matrix)):
            for j in range(len(self.task_matrix[i])):
                for k in range(len(self.task_matrix[i][j])):
                    self.task_matrix[i][j][k].index = indices
                    indices += 1

    def dnn_partitioning(self):

