import random
from NetworkImplementation import Network
from abc import abstractmethod, ABC
from TaskImplementation import Task
from Distributions import custom_distribution
from typing import Callable
from SatelliteImplementation import Satellite
import numpy as np


class Strategy(ABC):
    @abstractmethod
    def __init__(self, network: Network):
        pass

    @abstractmethod
    def decide_a_scheme_for(self, mcd: int, center_x: int, center_y: int,
                            task_block: list[Task], scheme_length: int) -> list[tuple]:
        pass
