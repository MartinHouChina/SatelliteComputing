import numpy as np
import math
from TaskImplementation import Task


def normal_distribution(mean: float, variance: float, size: None | int | tuple = None):
    """
    返回一个大小为 size 的高斯分布抽样结果，符合 X ~ N(mean, variance^2)
    """
    rng = np.random.default_rng()
    return rng.normal(mean, variance, size=size)


def poisson_distribution(lam: float, size: None | int | tuple = None) -> list[...] | float:
    """
    返回一个大小为 size 的泊松分布抽样结果，符合 X ~ P(lam)
    """
    rng = np.random.default_rng()
    return rng.poisson(lam, size=size)


def uniform_distribution(lower: float, upper: float, size: None | int | tuple = None) -> list[...] | float:
    """
    返回一个大小为 size 的均匀分布抽样结果，符合 X ~ U(lower, upper)
    """
    rng = np.random.default_rng()

    return rng.uniform(lower, upper, size=size)


def custom_distribution(distribution):
    """
    自定义分布，请确保你的分布与网络大小适配
    """
    return distribution


def summon_task_matrix(width: int, height: int, layer_cnt: int, lam: float, lower: float, upper: float):
    """
    生成符合特定要求的任务矩阵
    """
    task_matrix = [[list() for __ in range(height)] for _ in range(width)]
    task_cnt_matrix = poisson_distribution(lam, (width, height))
    for i in range(width):
        for j in range(height):
            cnt = math.ceil(task_cnt_matrix[i][j])
            for _ in range(cnt):
                task = Task(uniform_distribution(lower, upper, layer_cnt))
                task_matrix[i][j].append(task)

    return task_matrix


if __name__ == '__main__':
    task_matrix = summon_task_matrix(2, 2, 5, 4, 1, 5)
    print(task_matrix, '\n', task_matrix[0], '\n',
          task_matrix[0][0], '\n', task_matrix[0][0][0].layers)
