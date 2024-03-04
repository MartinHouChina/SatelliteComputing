from TaskImplementation import Task
from queue import Queue
from collections import defaultdict


class Satellite:
    def __init__(self, capability):
        self.max_capability = capability
        self.capability = capability
        self.processing_list = defaultdict(lambda: Queue())  # 正在接受处理的任务字典
        self.completed_cnt = 0
        self.abandon_cnt = 0

    def load(self, slices: list[Task]):
        """
        装载任务块到 CPU 上, 同时会将执行状况记录后指派，返回成功处理的工作量与列表，丢包列表
        """
        slices.sort(key=lambda x: x.total_workload)
        successful_workload, successful_list, drop_list = 0, [], []
        for idx, slice in enumerate(slices):
            workload = slice.total_workload
            slice_idx = slice.idx
            if workload < self.capability:
                self.capability -= workload
                self.completed_cnt += 1
                successful_workload += workload
                successful_list.append(slice.idx)
                self.processing_list[slice_idx].put(workload)
            else:
                drop_list = [task.idx for task in slices[idx:]]
                self.abandon_cnt += len(slices) - idx
                break
        return successful_workload, successful_list, drop_list

    def offload(self, task_index: int):
        """
        卸载某一接受处理的任务，如果该任务不存在则报错。
        """
        if task_index not in self.processing_list:
            raise RuntimeError("No such index!", task_index,
                               self.processing_list)
        else:
            self.capability += self.processing_list[task_index].get()

    def reset(self):
        self.capability = self.max_capability
        self.processing_list.clear()
        self.completed_cnt = 0
        self.abandon_cnt = 0
