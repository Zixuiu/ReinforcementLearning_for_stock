import os
import yaml
from stable_baselines3 import PPO, A2C, DDPG, TD3
from util import find_file, prepare_env

with open('config.yaml') as f:
    args = yaml.safe_load(f)

def train_model(env, RL_model='PPO'):
    # 根据所选择的RL模型创建对应的模型对象
    if RL_model == 'A2C':
        model = A2C("MlpPolicy", env, verbose=0, tensorboard_log='./log')
    elif RL_model == 'PPO':
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log='./log')
    elif RL_model == 'DDPG':
        model = DDPG("MlpPolicy", env, verbose=0, tensorboard_log='./log')
    elif RL_model == 'TD3':
        model = TD3("MlpPolicy", env, verbose=0, tensorboard_log='./log')

    # 使用指定的环境和模型开始训练
    model.learn(total_timesteps=args['train_args']['total_timesteps'])

    return model

def train_strategy(stock_code, RL_model):
    # 获得股票文件路径
    stock_file = find_file('./data/tushare_data/train', str(stock_code))
    # 准备环境
    train_env, _ = prepare_env(stock_file)
    # 训练模型
    model = train_model(train_env, RL_model)

    # 创建保存模型的文件夹
    os.makedirs('./check_points/', exist_ok=True)
    # 保存模型
    model.save(f"./check_points/{RL_model}_{stock_code}")

if __name__ == '__main__':
    # 从配置文件中读取参数
    train_strategy(args['train_args']['stock_code'], args['train_args']['rl_model'])
