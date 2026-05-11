import base64
import requests
import os


# 1. 将本地图片转换为大模型能看懂的 Base64 格式
def encode_image(image_path):
    if not os.path.exists(image_path):
        print(f"找不到图片: {image_path}")
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# 2. 调用大模型视觉接口 (以 SiliconFlow 为例)
def analyze_simulation_chart(image_path):
    base64_image = encode_image(image_path)
    if not base64_image: return

    # 请替换为你自己的 API KEY
    API_KEY = "sk-fbmbehwyzlkshcblcwlyekpgqrrmztzjsgoctmoaphxdgbee"

    # 使用通用的 OpenAI 兼容接口格式
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # 精心设计的系统提示词 (Prompt Engineering)
    prompt = """
    你是一个拥有十年经验的资深云计算架构师。
    这是一张深度强化学习(PPO算法)应用于云服务器资源调度的实验监控图。
    请你分析这张图表，并回答以下问题：
    1. 曲线反映出的模型收敛趋势如何？
    2. 在系统调度过程中，是否有明显的性能瓶颈或震荡？
    3. 请给出一到两条后续优化调度算法的建议。
    """

    payload = {
        "model": "Qwen/Qwen2-VL-72B-Instruct",  # 强大的视觉模型
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 1000
    }

    print("🧠 AI 架构师正在分析图像，请稍候...")
    try:
        response = requests.post("https://api.siliconflow.cn/v1/chat/completions", headers=headers, json=payload)
        result = response.json()
        print("\n" + "=" * 50)
        print("📊 大模型分析报告：")
        print(result['choices'][0]['message']['content'])
        print("=" * 50 + "\n")
    except Exception as e:
        print(f"调用 API 失败: {e}")


if __name__ == "__main__":
    # 测试一下：让它分析你仓库里的那张 reward_mean.png
    target_image = "assets/images/reward_mean.png"
    analyze_simulation_chart(target_image)