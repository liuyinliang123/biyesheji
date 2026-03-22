import os
import sys
import numpy as np

# ================= 路径修复工具 =================
current_path = os.path.abspath(os.path.dirname(__file__))
if current_path not in sys.path:
    sys.path.insert(0, current_path)
# ===============================================

try:
    import gym
    import envs  # 注册自定义环境
    from stable_baselines3 import PPO
    from utils import calculate_average_slowdown  #
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)

# 核心配置
ENV_ID = 'deepcss-v2'
MODEL_PATH = 'logs/my_model_v4_2M/best_model.zip'  # 确认路径正确
ITERS = 20


def count_completed_jobs(info):
    """
    统计在该回合中真正完成（finish_time > 0）的任务数量
    """
    job_rec = info['job_record'].record
    completed_count = 0
    for index in job_rec:
        if job_rec[index].finish_time > 0:
            completed_count += 1
    return completed_count


def run_sjf_strategy(env):
    """SJF 策略：选择最短任务"""
    job_slot = env.job_slot.slot
    num_wq = env.pa.num_wq
    shortest_job_idx = None
    min_len = float('inf')
    for idx, job in enumerate(job_slot):
        if job is not None and job.len < min_len:
            min_len = job.len
            shortest_job_idx = idx
    if shortest_job_idx is None:
        return [num_wq, 0, 0]
    return [shortest_job_idx, 0, 0]


def evaluate():
    print(f"正在启动深度对比实验: {ENV_ID} (测试轮数: {ITERS})")
    env = gym.make(ENV_ID)
    try:
        model = PPO.load(MODEL_PATH)
    except:
        print(f"无法加载模型: {MODEL_PATH}")
        return

    # 初始化统计数据
    metrics = {
        'Random': {'rewards': [], 'slowdowns': [], 'completed': []},
        'SJF': {'rewards': [], 'slowdowns': [], 'completed': []},
        'AI (PPO)': {'rewards': [], 'slowdowns': [], 'completed': []}
    }

    for i in range(ITERS):
        for name in metrics.keys():
            obs = env.reset()
            done, ep_rew = False, 0
            while not done:
                if name == 'Random':
                    action = env.action_space.sample()
                elif name == 'SJF':
                    action = run_sjf_strategy(env)
                else:  # AI
                    action, _ = model.predict(obs, deterministic=True)

                obs, rew, done, info = env.step(action)
                ep_rew += rew

            # 记录数据
            metrics[name]['rewards'].append(ep_rew)
            metrics[name]['slowdowns'].append(calculate_average_slowdown(info))
            metrics[name]['completed'].append(count_completed_jobs(info))

        if (i + 1) % 5 == 0: print(f"进度: {i + 1}/{ITERS}")

    print("\n" + "=" * 75)
    print(f"{'调度策略':<15} | {'平均奖励':<12} | {'周转时间(越小越好)':<18} | {'任务完成数(吞吐量)'}")
    print("-" * 75)
    for name in metrics.keys():
        avg_r = np.mean(metrics[name]['rewards'])
        avg_s = np.mean(metrics[name]['slowdowns'])
        avg_c = np.mean(metrics[name]['completed'])
        print(f"{name:<15} | {avg_r:<12.2f} | {avg_s:<18.2f} | {avg_c:.1f}")
    print("=" * 75)



if __name__ == "__main__":
    evaluate()