import os
import numpy as np
import gym
from gym import spaces
from .parameters import Parameters


# =================================================================
# 核心类：Env - 模拟云服务器调度环境，遵循 OpenAI Gym 标准接口
# =================================================================
class Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, pa, render=False,
                 repre='compact', end='all_done', seed=33) -> None:
        """
        初始化环境
        :param pa: 参数配置对象 (Parameters)
        :param render: 是否实时渲染训练/评估画面
        :param repre: 状态表示方式 ('compact' 为精简向量模式)
        :param end: 回合结束条件 ('all_done' 表示所有任务处理完才结束)
        :param seed: 随机种子
        """
        super(Env, self).__init__()

        self.pa = pa
        self.env_seed = seed
        if not self.pa.unseen:
            np.random.seed(self.env_seed)

        # 定义动作空间：[任务槽索引, 优先级, 服务器编号]
        # 任务槽索引范围是 0 到 num_wq (最后一个索引代表“跳过/不操作”)
        self.action_space = spaces.MultiDiscrete([self.pa.num_wq + 1,
                                                  self.pa.num_prio,
                                                  self.pa.num_serv])

        # 定义观测空间（状态空间）：包含服务器状态、等待队列状态等
        if repre == 'compact':
            self.observation_space = spaces.Box(low=0,
                                                high=1,
                                                shape=(self.pa.time_horizon * self.pa.num_serv +
                                                       self.pa.time_horizon * self.pa.num_serv +
                                                       self.pa.num_wq + 2,),
                                                dtype=np.float64)

        self.env_render = render
        self.repre = repre  # 图像表示或精简向量表示
        self.end = end  # 终止类型：'no_new_job' (无新任务) 或 'all_done' (全部完成)

        self.curr_time = 0
        # 任务长度遵循指数分布 (Exponential Distribution)
        self.work_dist = self.pa.work_dist.exp_model_dist

        # 预先生成整个回合的任务序列
        self.work_len_seqs = self.generate_work_sequence(
            simu_len=self.pa.simu_len)
        self.work_len_seqs.insert(0, [])  # 初始时刻通常没有任务
        self.env_len = len(self.work_len_seqs)

        self.seq_idx = 0
        # 初始化系统各大组件
        self.machine = Machine(pa)  # 物理服务器集群
        self.job_record = self.machine.fill_up_servers()  # 初始状态填充部分背景任务
        self.job_slot = JobSlot(pa)  # 可见的待处理任务槽
        self.job_backlog = JobBacklog(pa)  # 缓冲区满后的积压区
        self.extra_info = ExtraInfo(pa)  # 额外的时间信息追踪

    def step(self, a):
        """
        环境的核心推进逻辑：执行一个动作，返回新状态、奖励、是否结束和信息
        """
        action = a[0]  # 选择哪个任务槽的任务
        priority = a[1]  # 分配什么优先级
        server = a[2]  # 分配到哪台服务器
        status = None
        done = False
        reward = 0
        info = None

        # 逻辑判断：如果选择的是最后一个索引，或者选中的槽位是空的
        if action == self.pa.num_wq:  # 显式空动作：Agent主动选择不调度
            status = 'MoveOn'
        elif self.job_slot.slot[action] is None:  # 隐式空动作：Agent选了个空槽
            status = 'MoveOn'
        else:
            # 尝试分配任务
            self.job_slot.slot[action].priority = priority
            allocated = self.machine.allocate_job(self.job_slot.slot[action],
                                                  server)
            # 如果服务器资源够，则分配成功；否则强制进入“时间推进”模式
            status = 'Allocate' if allocated else 'MoveOn'

        # 如果当前没有动作可以执行，时间向前推进一格
        if status == 'MoveOn':
            self.curr_time += 1
            self.machine.time_proceed(self.curr_time)  # 运行中的任务剩余时间减1
            self.extra_info.time_proceed()

            # 尝试从预生成的序列中添加新任务
            self.seq_idx += 1
            if self.end == 'no_new_job':
                if self.seq_idx >= self.env_len:
                    done = True
            elif self.end == "all_done":  # 必须等所有任务都处理完
                if self.seq_idx >= self.env_len and \
                        self.all_servers_empty() and \
                        all(s is None for s in self.job_slot.slot) and \
                        all(s is None for s in self.job_backlog.backlog):
                    done = True
                elif self.curr_time > self.pa.episode_max_length:  # 运行太久，强制终止
                    done = True

            if not done:
                if self.seq_idx < self.env_len:
                    new_jobs = self.get_new_job_from_seq(self.seq_idx)

                    for job in new_jobs:  # 新任务到达
                        to_backlog = True

                        # 优先填入可见的任务槽 (Job Slot)
                        for i in range(self.pa.num_wq):
                            if self.job_slot.slot[i] is None:
                                self.job_slot.slot[i] = job
                                to_backlog = False
                                break

                        # 如果任务槽满了，进入积压区 (Backlog)
                        if to_backlog:
                            if self.job_backlog.curr_size < self.pa.backlog_size:
                                self.job_backlog.backlog[self.job_backlog.curr_size] = job
                                self.job_backlog.curr_size += 1
                            else:  # 积压区也满了，任务被迫丢失
                                print("积压区已满，任务丢失。")
                                del self.job_record.record[job.id]

                        self.extra_info.new_job_comes()
            reward = self.get_reward()

        elif status == 'Allocate':
            # 分配成功：从任务槽移除该任务
            self.job_record.record[self.job_slot.slot[action].id] = self.job_slot.slot[action]
            self.job_slot.slot[action] = None

            # 从积压区取出第一个任务填补刚才空出来的槽位 (FIFO 队列)
            if self.job_backlog.curr_size > 0:
                self.job_slot.slot[action] = self.job_backlog.backlog[0]
                self.job_backlog.backlog[:-1] = self.job_backlog.backlog[1:]
                self.job_backlog.backlog[-1] = None
                self.job_backlog.curr_size -= 1

        ob = self.observe()  # 获取新状态
        info = self.job_record

        if self.env_render:
            self.plot_state()  # 渲染画面

        information = {}
        information['job_record'] = info
        return ob, reward, done, information

    def get_reward(self):
        """
        奖励函数：定义 Agent 的目标。由于是调度问题，奖励通常是负值（惩罚）
        """
        reward = 0
        # 惩罚1：正在服务器中运行的任务产生的延迟（任务越长惩罚越轻）
        for serv in self.machine.running_jobs:
            for k in serv:
                for job in serv[k]:
                    reward += self.pa.delay_penalty / float(job.len)

        # 惩罚2：任务待在任务槽中未被调度
        for job in self.job_slot.slot:
            if job is not None:
                reward += self.pa.hold_penalty / float(job.len)

        # 惩罚3：任务待在积压区中
        for job in self.job_backlog.backlog:
            if job is not None:
                reward += self.pa.dismiss_penalty / float(job.len)

        return reward

    def observe(self):
        """
        状态观测：将复杂的系统状态转化为神经网络能理解的 [0, 1] 之间的浮点向量
        """
        if self.repre == "compact":
            compact_repre = np.zeros((self.pa.time_horizon * self.pa.num_serv) +  # 服务器任务状态
                                     (self.pa.time_horizon * self.pa.num_serv) +  # 服务器优先级状态
                                     self.pa.num_wq +  # 任务队列状态
                                     2,  # 积压区和额外信息
                                     dtype=np.float64)
            running_jobs = self.machine.running_jobs
            job_slot = self.job_slot.slot
            backlog_curr_size = self.job_backlog.curr_size
            backlog_size = self.pa.backlog_size
            extra_info = self.extra_info.time_since_last_new_job

            work_queue = np.zeros((self.pa.num_wq, 1), dtype=np.float64)
            srv_works = np.zeros((self.pa.time_horizon, self.pa.num_serv), dtype=np.float64)
            srv_prios = np.zeros((self.pa.time_horizon, self.pa.num_serv), dtype=np.float64)

            # 遍历服务器，填充当前运行任务的归一化剩余时间和优先级
            for idx, serv in enumerate(running_jobs):
                ptr = 0
                for prio in serv:
                    for job in serv[prio]:
                        srv_works[ptr, idx] += job.remaining_time / self.pa.max_job_len
                        srv_prios[ptr, idx] += (job.priority + 1) / self.pa.num_prio
                        ptr += 1

            # 填充待处理任务队列状态
            for idx, job in enumerate(job_slot):
                if job is not None:
                    work_queue[idx] = job.len / self.pa.max_job_len

            ptr = 0
            srv_works = srv_works.flatten('F')  # 按列展开
            srv_prios = srv_prios.flatten('F')
            work_queue = work_queue.flatten()

            # 拼接所有信息到最终的向量中
            compact_repre[ptr: srv_works.shape[0]] = srv_works
            ptr += srv_works.shape[0]
            compact_repre[ptr: ptr + srv_prios.shape[0]] = srv_prios
            ptr += srv_prios.shape[0]
            compact_repre[ptr: ptr + work_queue.shape[0]] = work_queue
            ptr += work_queue.shape[0]
            compact_repre[ptr] = backlog_curr_size / backlog_size  # 积压区占用率
            compact_repre[ptr + 1] = extra_info / self.extra_info.max_tracking_time_since_last_job  # 任务到达间隔
            return compact_repre

    def render(self):
        pass

    def all_servers_empty(self):
        """检查所有服务器是否都已经空闲"""
        empty = True
        for dic in self.machine.running_jobs:
            for k in dic:
                if dic[k]:
                    empty = False
                    break
            if not empty:
                break
        return empty

    def generate_work_sequence(self, simu_len):
        """
        生成任务序列：使用泊松分布 (Poisson) 决定每个时刻到达的任务数量
        """
        work_len_seq = []
        size = 0
        while size < simu_len:
            cnt = np.random.poisson(self.pa.lamda)  # 泊松分布产生新任务数
            if cnt + size > simu_len:
                continue
            size += cnt
            work_len_seq.append([self.work_dist() for _ in range(cnt)])
        return work_len_seq

    def get_new_job_from_seq(self, seq_index):
        """从预生成的序列中获取具体时刻的新任务对象"""
        jobs = []
        for l in self.work_len_seqs[seq_index]:
            new_job = Job(job_id=len(self.job_record.record),
                          job_len=l,
                          enter_time=self.curr_time)
            self.job_record.record[new_job.id] = new_job
            jobs.append(new_job)
        return jobs

    def plot_state(self):
        """控制台打印当前的系统状态动画"""
        os.system('cls')  # Windows 清屏命令
        for idx, serv in enumerate(self.machine.running_jobs):
            print(f"服务器编号: {idx}")
            for k in serv:
                print(f"\t优先级: {k}")
                for job in serv[k]:
                    if job is not None:
                        print(f"\t\t任务 ID: ", job.id, end='')
                        print(f"\t剩余时间: ", job.remaining_time)
        for job in self.job_slot.slot:
            if job is not None:
                print("任务 ID: ", job.id, end='')
                print(f"\t任务长度: ", job.remaining_time)
            else:
                print("空 (None)")

        print("当前积压区大小: ", self.job_backlog.curr_size)
        print("距离上个任务到达已过去时间: ",
              self.extra_info.time_since_last_new_job)

    def reset(self):
        """重置环境：当一局结束或开始时调用"""
        self.seq_idx = 0
        self.curr_time = 0

        if not self.pa.unseen:
            np.random.seed(self.env_seed)

        self.work_len_seqs = self.generate_work_sequence(simu_len=self.pa.simu_len)
        self.work_len_seqs.insert(0, [])
        self.env_len = len(self.work_len_seqs)

        # 重新初始化系统状态
        self.machine = Machine(self.pa)
        self.job_record = self.machine.fill_up_servers()
        self.job_slot = JobSlot(self.pa)
        self.job_backlog = JobBacklog(self.pa)
        self.extra_info = ExtraInfo(self.pa)

        return self.observe()

    def close(self) -> None:
        return super().close()


# =================================================================
# 辅助类：Job - 定义任务的数据结构
# =================================================================
class Job:
    """
    单个任务的结构
    id : 任务唯一标识
    len : 任务总时长
    enter_time : 进入队列的时间步
    start_time : 开始执行的时间步
    finish_time : 任务结束的时间步
    priority : 优先级 (0 为最高)
    remaining_time : 距离任务完成还剩多少时间步
    """

    def __init__(self, job_len, job_id, enter_time, priority=-1) -> None:
        self.id = job_id
        self.len = job_len
        self.enter_time = enter_time
        self.start_time = -1
        self.finish_time = -1
        self.priority = priority
        self.remaining_time = job_len

    def __str__(self) -> str:
        return f"id={self.id}, len={self.len}, enter_time={self.enter_time}, prio={self.priority}, remain={self.remaining_time}"


# =================================================================
# 辅助类：JobSlot, JobBacklog, JobRecord, ExtraInfo
# =================================================================
class JobSlot:
    """可见任务槽：Agent 只能从这里选择任务进行调度"""

    def __init__(self, pa) -> None:
        self.slot = [None] * pa.num_wq


class JobBacklog:
    """积压区：当任务槽满时，新任务在这里排队"""

    def __init__(self, pa) -> None:
        self.backlog = [None] * pa.backlog_size
        self.curr_size = int(0)


class JobRecord:
    """历史记录：存储模拟过程中出现过的所有任务"""

    def __init__(self) -> None:
        self.record = {}


class ExtraInfo:
    """追踪任务到达的时间间隔"""

    def __init__(self, pa):
        self.time_since_last_new_job = 0
        self.max_tracking_time_since_last_job = pa.max_track_since_new

    def new_job_comes(self):
        self.time_since_last_new_job = 0

    def time_proceed(self):
        if self.time_since_last_new_job < self.max_tracking_time_since_last_job:
            self.time_since_last_new_job += 1


# =================================================================
# 核心辅助类：Machine - 处理服务器的物理逻辑
# =================================================================
class Machine:
    def __init__(self, pa: Parameters) -> None:
        self.num_serv = pa.num_serv
        self.time_horizon = pa.time_horizon
        self.crowded_p = pa.crowded_p
        self.min_crowded = pa.min_crowded_congestion
        self.max_crowded = pa.max_crowded_congestion
        self.max_uncrowded = pa.max_uncrowded_congestion
        self.num_prio = pa.num_prio

        # 每台服务器剩余的可用容量（时间片槽位）
        self.avlbl_slots = np.array([self.time_horizon for _ in range(self.num_serv)])
        # 每台服务器中正在运行的任务列表（按优先级分类）
        self.running_jobs = [{prio: [] for prio in range(pa.num_prio)} for _ in range(self.num_serv)]

    def fill_up_servers(self):
        """
        环境初始化时，随机为服务器填充一些初始任务，模拟“拥挤”的机房
        """
        job_record = JobRecord()
        crowded = False
        for idx in range(self.num_serv):
            # 以一定概率决定该服务器是否拥挤
            if np.random.random() < self.crowded_p:
                crowded = True
                congestion = np.random.randint(self.min_crowded, self.max_crowded + 1)
            else:
                congestion = np.random.randint(0, self.max_uncrowded + 1)

            avlbl_slots = self.time_horizon - congestion

            while self.avlbl_slots[idx] > avlbl_slots:
                work_len = np.random.randint(1, self.max_crowded) if crowded \
                    else np.random.randint(1, self.max_uncrowded)
                if self.avlbl_slots[idx] - work_len < avlbl_slots:
                    continue
                prio = np.random.randint(1, self.num_prio)
                new_job = Job(job_id=len(job_record.record),
                              job_len=work_len,
                              enter_time=0,
                              priority=prio)
                job_record.record[new_job.id] = new_job
                self.avlbl_slots[idx] -= work_len
                self.running_jobs[idx][prio].append(new_job)
        return job_record

    def allocate_job(self, job: Job, num_serv):
        """
        尝试将任务分配到指定的服务器。
        如果服务器剩余空间足够，返回 True，否则返回 False。
        """
        allocate = False
        prio = job.priority
        if job.len <= self.avlbl_slots[num_serv]:
            allocate = True
            self.avlbl_slots[num_serv] -= job.len
            self.running_jobs[num_serv][prio].append(job)
        return allocate

    def time_proceed(self, curr_time):
        """
        模拟时间推移：
        1. 服务器中所有正在运行任务的剩余时间减 1。
        2. 如果任务完成，释放服务器资源。
        """
        prev_time = curr_time - 1
        for idx in range(len(self.running_jobs)):
            serv = self.running_jobs[idx]
            for k in sorted(serv):  # 按优先级顺序处理
                if serv[k]:
                    job = serv[k].pop(0)
                    if job.start_time == -1:
                        job.start_time = prev_time
                    if job.remaining_time > 0:
                        job.remaining_time -= 1
                        self.avlbl_slots[idx] += 1
                    if job.remaining_time == 0:
                        job.finish_time = curr_time
                    else:
                        serv[k].insert(0, job)  # 没做完，塞回去继续
                    break  # 每个时刻每台服务器只推进一个任务的工作量