import gym
import envs
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.utils import get_schedule_fn, set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

from utils import calculate_average_slowdown, SJF, make_env
from arg_loader import arg_parser
from envs.deepcss_v0.environment import Parameters as V0_PA
from envs.deepcss_v1.environment import Parameters as V1_PA
from envs.deepcss_v2.environment import Parameters as V2_PA


def make_eval_envs(pa):
    eval_envs = []
    for seed in pa.eval_seeds:
        eval_envs.append(make_env(seed=seed))
    eval_envs = DummyVecEnv(eval_envs)
    return eval_envs


def main():
    args = arg_parser()
    ENV_ID = args.environment
    TIMESTEPS = args.timesteps
    CPU = args.cpu
    CLIP_FN = get_schedule_fn(args.cliprange)

    vec_envs = make_vec_env(ENV_ID, n_envs=CPU)
    eval_envs = make_vec_env(ENV_ID, n_envs=10)
    if ENV_ID == 'deepcss-v0':
        policy_kwargs = V0_PA().policy_kwargs
    elif ENV_ID == 'deepcss-v1':
        policy_kwargs = V1_PA().policy_kwargs
    elif ENV_ID == 'deepcss-v2':
        policy_kwargs = V2_PA().policy_kwargs
    else:
        print('Environment ID not found.')
        exit(0)

    if args.mode == 'train':
        if not args.name:
            print("Please set a name for training model")
            exit(0)
        else:
            MODEL_NAME = args.name
        checkpoint_callback = CheckpointCallback(save_freq=5_000,
                                                 save_path=f'./models/{MODEL_NAME}_{args.algorithm}',
                                                 name_prefix=f'{args.algorithm}')
        eval_callback = EvalCallback(eval_envs, best_model_save_path=f'./logs/{MODEL_NAME}/',
                                     log_path=f'./logs/{MODEL_NAME}/', eval_freq=2_500, deterministic=True,
                                     render=False)
        callbacks = CallbackList([checkpoint_callback, eval_callback])

        if args.algorithm == 'ppo':
            print('creating model')
            model = PPO('MlpPolicy', vec_envs, batch_size=args.batchsize,
                        tensorboard_log='./tensorboard/', device='auto',
                        clip_range=CLIP_FN, policy_kwargs=policy_kwargs)
            if args.load:
                print(f"loading model from: {args.load}")
                model = model.load(args.load, vec_envs)
                model.clip_range = CLIP_FN
            try:
                print(f"training on {model.device}")
                model.learn(TIMESTEPS, callback=callbacks,
                            tb_log_name=f'{MODEL_NAME}_{args.algorithm}')
            except:
                model.save(f'tmp/{MODEL_NAME}')
                print(f"model trained using {args.algorithm} algorithm.")
                exit(0)
        elif args.algorithm == 'dqn':
            pass

    # 修复了这里的缩进，并增加了评估结果打印
    elif args.mode == 'eval':
        if not args.load:
            print("model path not specified (--load model_path)")
            exit(0)
        print(f"正在加载模型: {args.load} ...")
        eval_env = gym.make(ENV_ID,render=True)
        model = PPO.load(args.load, env=eval_env)

        print(f"开始评估，共测试 {args.iters} 轮...")
        # 接收评估结果
        mean_reward, std_reward = evaluate_policy(model, eval_env,
                                                  n_eval_episodes=args.iters,
                                                  deterministic=True)

        # 打印结果
        print("-" * 30)
        print(f"评估环境: {ENV_ID}")
        print(f"平均奖励 (Mean Reward): {mean_reward:.2f}")
        print(f"奖励标准差 (Std Reward): {std_reward:.2f}")
        print("-" * 30)


if __name__ == '__main__':
    main()