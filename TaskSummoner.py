import numpy as np


def normal_distribution(mean: float, variance: float, size: None | int | tuple = None):
    """
    返回一个大小为 size 的高斯分布抽样结果，符合 X ~ N(mean, variance^2)
    """
    rng = np.random.default_rng()
    return rng.normal(mean, variance, size=size)


def poisson_distribution(lam: float, size: None | int | tuple = None):
    """
    返回一个大小为 size 的泊松分布抽样结果，符合 X ~ P(lam)
    """
    rng = np.random.default_rng()
    return rng.poisson(lam, size=size)


def uniform_distribution(lower: float, upper: float, size: None | int | tuple = None):
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
