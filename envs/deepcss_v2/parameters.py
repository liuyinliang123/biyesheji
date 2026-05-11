import numpy as np
from . import job_distribution


# =================================================================
# 类名：Parameters - 环境全局参数配置中心
# 作用：集中管理模拟长度、服务器规格、奖励权重及神经网络层数
# =================================================================
class Parameters:
    def __init__(self) -> None:
        # --- 仿真控制参数 ---
        # 仿真长度：每次模拟中进入系统的总任务数量
        self.simu_len = 30
        # 最大回合长度：一个回合允许运行的最大时间步数
        self.episode_max_length = 500

        # --- 服务器集群规格 ---
        # 服务器数量：系统内可用的物理服务器总数
        self.num_serv = 3
        # 可见队列大小：Agent 能够直接看到的待处理任务槽（堆区）数量
        self.num_wq = 10
        # 优先级层级：任务可以被分配的不同优先级数量（0 为最高）
        self.num_prio = 3

        # --- 任务属性设定 ---
        # 时间跨度：每台服务器的总容量/总时间槽位
        self.time_horizon = 20
        # 任务最大长度：单个新产生任务允许的最大执行时长
        self.max_job_len = 15

        # --- 缓冲区与追踪 ---
        # 积压区大小：当可见队列满时，用于存放排队任务的缓冲区容量
        self.backlog_size = 60
        # 任务间隔追踪：追踪自上一个新任务到达以来经过的最大时间步数
        self.max_track_since_new = 10

        # --- 任务到达概率模型 (泊松分布相关) ---
        # 新任务到达率：泊松过程中新任务到达的概率因子
        self.new_job_rate = 0.8
        # 泊松分布 Lambda 值：每个时间步平均到达的任务数量
        self.lamda = 3
        # 单步最大任务数：一个时间步内允许产生的最大新任务数量
        self.max_job_cnt = 5

        # 是否使用不可见种子：决定使用固定种子还是随机种子（True 为随机）
        self.unseen = True

        # 任务长度分布：调用 job_distribution 生成符合特定规律的任务时长
        self.work_dist = job_distribution.Dist(self.max_job_len)

        # --- 奖励函数权重 (均为负值，代表惩罚) ---
        # 等待惩罚：任务留在可见队列中未被调度时的惩罚
        self.hold_penalty = -1
        # 丢弃惩罚：任务在积压区中堆积产生的惩罚
        self.dismiss_penalty = -1
        # 延迟惩罚：任务在服务器中运行过程中产生的延迟惩罚
        self.delay_penalty = -1

        # --- 服务器初始拥堵状态参数 ---
        # ----------------------------
        # 服务器“拥挤”概率：初始化时，某台服务器处于高负载状态的概率
        self.crowded_p = 0.3
        # 拥挤服务器最大负载：处于拥挤状态时，预填任务占用的最大槽位数
        self.max_crowded_congestion = 15
        # 拥挤服务器最小负载：处于拥挤状态时，预填任务占用的最小槽位数
        self.min_crowded_congestion = 10
        # 非拥挤服务器最大负载：处于正常状态时，允许存在的最大预填任务数
        self.max_uncrowded_congestion = 5
        # ----------------------------

        # --- 神经网络架构配置 ---
        # Value Network (价值网络)：Critic 网络每层神经元的数量
        self.vf_net = [256, 256, 256, 256]
        # Policy Network (策略网络)：Actor 网络每层神经元的数量
        self.pi_net = [256, 256, 256, 256]

        # 策略参数打包：用于 Stable-baselines3 框架初始化模型
        self.policy_kwargs = {
            "net_arch": [{
                "vf": self.vf_net,
                "pi": self.pi_net
            }]
        }