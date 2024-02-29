from NetworkImplementation import Network
from TaskImplementation import Task
from StrategyImplementation import Strategy


class Event:
    def __init__(self, center_x: int, center_y: int, task_block: list[list[Task]]
                 , cur_idx: int, dfn: float, network: Network):
        self.trajectory = None
        self.center_x = center_x
        self.center_y = center_y
        self.cur_pos = cur_idx
        self.dfn = dfn
        self.total_time = 0
        self.total_drop = 0
        self.total_successful = 0
        self.status = "undetermined"
        self.network = network
        self.task_block = task_block
        self.abandon_set = set()
        self.cache = tuple()

    def __lt__(self, other):
        return self.dfn < other.dfn

    def determine_with_strategy(self, strategy: Strategy, mcd: int, scheme_length: int):
        if self.status == "undetermined":
            self.trajectory = strategy.decide_a_scheme_for(mcd, self.center_x, self.center_y, self.task_block,
                                                           scheme_length)
            self.status = "load"

    def execute(self):
        network = self.network
        task_block = self.task_block

        if self.status == "undetermined":
            raise RuntimeError("An event of undetermined sequence!!!")

        if self.status == "load":
            # 目标卫星在网络中的位置
            target_x, target_y = self.trajectory[self.cur_pos]

            # 获取对应的任务块切片
            task_block_slice = [task[self.cur_pos] for task in task_block if
                                task[self.cur_pos].idx not in self.abandon_set]

            # 对应执行
            successful_workload, successful_list, drop_list = network.assign_with(target_x, target_y, task_block_slice)

            # 将丢包的任务剔除
            for ele in drop_list:
                self.abandon_set.add(ele)

            self.total_drop += len(drop_list)

            # 保存 successful_list, 为下一阶段的 offload 做准备
            self.cache = (successful_list, successful_workload)

            self.total_time += successful_workload

            # 后续 offload 状态更新
            self.dfn += successful_workload
            self.status = "offload"

        elif self.status == "offload":
            # 目标卫星在网络中的位置
            target_x, target_y = self.trajectory[self.cur_pos]

            # 获取对应的任务块切片
            task_block_slice = [task[self.cur_pos] for task in task_block]

            last_successful_list, last_successful_workload = self.cache

            # 撤销对应执行
            for idx in last_successful_list:
                network.undo_with(target_x, target_y, idx)

            # 清空缓存
            self.cache = tuple()

            if self.cur_pos + 1 != len(self.trajectory):
                # 后续 offload 状态更新
                nxt_x, nxt_y = self.trajectory[self.cur_pos + 1]
                transition_time = network.transition_cof_between(target_x, target_y, nxt_x,
                                                                 nxt_y) * last_successful_workload
                self.dfn += transition_time
                self.total_time += transition_time
                self.status = "load"
                self.cur_pos += 1

            else:
                self.total_successful = len(last_successful_list)
                self.status = "finished"
