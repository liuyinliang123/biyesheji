import os
import numpy as np
import matplotlib.pyplot as plt

# 1. 确保 asset 文件夹存在
save_path = 'assets/images'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 2. 读取你之前的评估数据 (从 logs 文件夹)
# 这里的路径根据你之前的截图设定
data_file = 'logs/my_model_v4_2M/evaluations.npz'

try:
    data = np.load(data_file)
    timesteps = data['timesteps']
    # 计算多次评估的平均分
    results = np.mean(data['results'], axis=1)

    # 3. 使用 matplotlib 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, results, color='#FF7F0E', label='PPO Reward')

    # 美化图片
    plt.title('Training Reward Analysis')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.grid(True, linestyle='--')
    plt.legend()

    # 4. 核心步骤：直接保存到 asset 文件夹
    file_name = os.path.join(save_path, 'ppo_training_curve.png')
    plt.savefig(file_name, dpi=300)  # dpi=300 保证图片非常清晰，适合打印论文
    print(f"✅ 图片已成功保存至: {file_name}")

except Exception as e:
    print(f"❌ 保存失败: {e}")