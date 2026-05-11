from typing import Callable
from envs.deepcss_v0.environment import Job, Env
from envs.deepcss_v0.parameters import Parameters
from stable_baselines3.common.monitor import Monitor


# =================================================================
# 函数 1：计算平均减速比 (Average Slowdown)
# 物理意义：这是评价调度算法好坏的核心指标。
# 计算公式：(完成时间 - 到达时间) / 任务长度。
# 值越小，说明任务在系统里等待的时间越短，用户体验越好。
# =================================================================
def calculate_average_slowdown(info):
    job_rec = info['job_record'].record  # 获取本回合所有任务的记录字典
    average_slowdown = 0
    len_count = 0
    for index in job_rec:
        len_count += 1
        job = job_rec[index]
        # 计算单个任务的减速比：周转时间 / 运行时间
        # job.finish_time - job.enter_time 即为该任务在系统中的总停留时间
        average_slowdown += (job.finish_time - job.enter_time) / job.len

    # 计算所有任务的平均值
    average_slowdown /= len_count
    return average_slowdown


# =================================================================
# 函数 2：SJF (Shortest Job First) 短作业优先算法
# 物理意义：这是一种经典的启发式调度算法。
# 逻辑：每次都从等待队列里挑出那个“活最少、干得最快”的任务先做。
# =================================================================
def SJF(env: Env):
    action = None
    min_len = 200  # 初始设为一个很大的数，用于寻找最小值

    # 遍历当前可见的任务槽 (Job Slots)
    for idx, job in enumerate(env.job_slot.slot):
        if job is not None:
            # 如果发现更短的任务，记录其索引
            if job.len < min_len:
                min_len = job.len
                action = idx

    # 如果没找到任何任务，返回 0（默认操作）
    if action is None:
        return 0
    return action


# =================================================================
# 函数 3：线性学习率调度 (Linear Learning Rate Schedule)
# 物理意义：训练初期学习率大（步子大，快速探索），训练后期学习率小（步子小，精细微调）。
# =================================================================
def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    创建一个线性递减的学习率函数。
    :param initial_value: 初始学习率 (例如 0.0003)
    :param final_value: 最终保底的学习率
    """

    def func(progress_remaining: float) -> float:
        """
        progress_remaining 会从 1.0 (训练开始) 逐渐减小到 0.0 (训练结束)
        """
        # 当训练进度还在前 75% 时，学习率随进度线性下降
        if progress_remaining > 0.25:
            return progress_remaining * initial_value
        else:
            # 当训练接近尾声（最后 25%）时，保持一个极小的稳定学习率
            return final_value

    return func


# =================================================================
# 函数 4：评估模型 (Evaluate Model)
# 作用：对比“随机策略”和“你的 AI 模型”在减速比指标上的表现。
# =================================================================
def eval_model(model, env, iters):
    methods = ['Random', 'Model Algorithm']
    random_mean = 0
    ml_mean = 0
    ITERS = iters

    for _ in range(ITERS):
        for m in methods:
            obs = env.reset()  # 重置环境
            while True:
                if m == 'Random':
                    action = env.action_space.sample()  # 随机瞎选动作
                else:
                    # 使用训练好的神经网络预测动作
                    action, _states = model.predict(obs, deterministic=True)

                obs, rewards, done, info = env.step(action)

                if done:  # 回合结束，记录该策略的平均减速比
                    if m == 'Random':
                        random_mean += calculate_average_slowdown(info)
                    else:
                        ml_mean += calculate_average_slowdown(info)
                    break

    # 打印对比结果
    print(f"随机策略平均减速比: {random_mean / ITERS :.2f}")
    print(f"AI模型平均减速比: {ml_mean / ITERS : .2f}")


# =================================================================
# 函数 5：创建环境辅助函数 (Make Environment)
# 作用：在并行训练时，为每个 CPU 核创建一个带监控器的环境实例。
# =================================================================
def make_env(seed):
    pa = Parameters()
    pa.unseen = False  # 设置为“已见过”的数据模式，方便稳定训练

    def _init():
        # Monitor 是 Stable Baselines 3 的工具，用于记录奖励、步数等统计信息
        env = Monitor(Env(pa, seed=seed))
        return env

    return _init