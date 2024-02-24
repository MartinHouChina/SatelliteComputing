from TaskImplementation import Task


class Satellite:
    def __init__(self, capability):
        self.capability = capability
        self.processing_list = dict()  # 正在接受处理的任务字典
        self.completed_cnt = 0
        self.abandon_cnt = 0
        self.mask = 0

    def load(self, slices: list[Task]):
        """
        装载任务块到 CPU 上, 同时会将执行状况记录后指派，返回成功处理的工作量与列表，丢包列表
        """
        slices.sort(key=lambda x: x.total_workload)
        output_slices = []
        successful_workload, successful_list, drop_list = 0, [], []
        self.mask = 0
        for idx, slice in enumerate(slices):
            workload = slice.total_workload
            slice_idx = slice.idx
            if workload < self.capability:
                self.capability -= workload
                self.completed_cnt += 1
                successful_workload += workload
                successful_list.append(slice.idx)
                self.processing_list[slice_idx] = workload
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
            return
        else:
            self.capability += self.processing_list[task_index]
            self.processing_list.pop(task_index)
