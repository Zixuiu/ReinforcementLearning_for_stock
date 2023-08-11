import time
import numpy as np
import pandas as pd

from env.MultiStock_train import StockEnvTrain
from env.MultiStock_validation import StockEnvValidation
from env.MultiStock_trade import StockEnvTrade

from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

# 设置数据路径
path = 'data/trading.csv'
df = pd.read_csv(path)

# 设置回测窗口和验证窗口的大小
rebalance_window = 63
validation_window = 63

# 获取唯一的交易日期
unique_trade_date = df[(df.datadate > 20151001) & (df.datadate <= 20200707)].datadate.unique()
print(unique_trade_date)

# 使用A2C算法进行训练
def train_A2C(env_train, model_name, timesteps=10):
    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"/Users/poteman/learn/RL/ReinforcementLearning_for_stock/archive/{model_name}")
    print(' - 训练时间 (A2C): ', (end - start) / 60, ' 分钟')
    return model

# 使用DDPG算法进行训练
def train_DDPG(env_train, model_name, timesteps=10):
    # 添加DDPG算法所需的噪声对象
    n_actions = env_train.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    start = time.time()
    model = DDPG('MlpPolicy', env_train, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"/Users/poteman/learn/RL/ReinforcementLearning_for_stock/archive/{model_name}")
    print(' - 训练时间 (DDPG): ', (end-start)/60,' 分钟')
    return model

# 使用PPO算法进行训练
def train_PPO(env_train, model_name, timesteps=50):
    start = time.time()
    model = PPO('MlpPolicy', env_train, ent_coef = 0.005)
    
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"/Users/poteman/learn/RL/ReinforcementLearning_for_stock/archive/{model_name}")
    print(' - 训练时间 (PPO): ', (end - start) / 60, ' 分钟')
    return model

# 将数据分为训练集和验证集
def data_split(df,start,end):
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data=data.sort_values(['datadate','tic'],ignore_index=True)
    data.index = data.datadate.factorize()[0]
    return data

# 获取验证集的夏普比率
def get_validation_sharpe(iteration):
    df_total_value = pd.read_csv('/Users/poteman/learn/RL/ReinforcementLearning_for_stock/archive/account_value_validation_{}.csv'.format(iteration), index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
    return sharpe

# 使用训练好的模型进行预测
def DRL_prediction(df,
                   model,
                   name,
                   last_state,
                   iter_num,
                   unique_trade_date,
                   rebalance_window,
                   turbulence_threshold,
                   initial):

    trade_data = data_split(df, start=unique_trade_date[iter_num - rebalance_window - validation_window], end=unique_trade_date[iter_num])
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                   turbulence_threshold=turbulence_threshold,
                                                   initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteration=iter_num)])
    obs_trade = env_trade.reset()

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2):
            last_state = env_trade.render()

    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('/Users/poteman/learn/RL/ReinforcementLearning_for_stock/archive/last_state_{}_{}.csv'.format(name, i), index=False)
    return last_state

# 在验证集上进行预测
def DRL_validation(model, test_data, test_env, test_obs) -> None:
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)

# 运行集成策略
def run_ensemble_strategy(df, unique_trade_date, rebalance_window, validation_window) -> None:
    last_state_ensemble = []
    ppo_sharpe_list = []
    ddpg_sharpe_list = []
    a2c_sharpe_list = []

    model_use = []

    insample_turbulence = df[(df.datadate<20151000) & (df.datadate>=20090000)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

    start = time.time()
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        if i - rebalance_window - validation_window == 0:
            # 初始状态
            initial = True
        else:
            # 上一个状态
            initial = False

        # 根据历史数据调整风险指数
        # 风险指数的回溯窗口为一个季度
        end_date_index = df.index[df["datadate"] == unique_trade_date[i - rebalance_window - validation_window]].to_list()[-1]
        start_date_index = end_date_index - validation_window*30 + 1

        historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])
        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        if historical_turbulence_mean > insample_turbulence_threshold:
            # 如果历史数据的均值大于样本内风险指数的90%分位数
            # 则认为当前市场波动较大
            # 因此将样本内风险指数的90%分位数作为风险指数阈值
            # 表示当前风险指数不能超过样本内风险指数的90%分位数
            turbulence_threshold = insample_turbulence_threshold
        else:
            # 如果历史数据的均值小于样本内风险指数的90%分位数
            # 则调高风险指数阈值，降低风险
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
            
        print("-" * 50)
        print(" - 风险指数阈值: ", turbulence_threshold)

        train = data_split(df, start=20090000, end=unique_trade_date[i - rebalance_window - validation_window])
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

        ## 验证集的StockEnv
        validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window],
                                end=unique_trade_date[i - rebalance_window])
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation,
                                                          turbulence_threshold=turbulence_threshold,
                                                          iteration=i)])
        obs_val = env_val.reset()
        
        print(" - 模型训练从: ", 20090000, "到 ",
              unique_trade_date[i - rebalance_window - validation_window])
        print(" - A2C 训练")
        model_a2c = train_A2C(env_train, model_name="A2C_30k_dow_{}".format(i), timesteps=30)
        print(" - A2C 验证从: ", unique_trade_date[i - rebalance_window - validation_window], "到 ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_a2c, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_a2c = get_validation_sharpe(i)
        print(" - A2C 夏普比率: ", sharpe_a2c)

        print(" - PPO 训练")
        model_ppo = train_PPO(env_train, model_name="PPO_100k_dow_{}".format(i), timesteps=10)
        print(" - PPO 验证从: ", unique_trade_date[i - rebalance_window - validation_window], "到 ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_ppo = get_validation_sharpe(i)
        print(" - PPO 夏普比率: ", sharpe_ppo)

        print(" - DDPG 训练")
        model_ddpg = train_DDPG(env_train, model_name="DDPG_10k_dow_{}".format(i), timesteps=10)
        print(" - DDPG 验证从: ", unique_trade_date[i - rebalance_window - validation_window], "到 ",
              unique_trade_date[i - rebalance_window])
        
        DRL_validation(model=model_ddpg, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_ddpg = get_validation_sharpe(i)

        ppo_sharpe_list.append(sharpe_ppo)
        a2c_sharpe_list.append(sharpe_a2c)
        ddpg_sharpe_list.append(sharpe_ddpg)

        # 根据夏普比率选择模型
        if (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_ddpg):
            model_ensemble = model_ppo
            model_use.append('PPO')
        elif (sharpe_a2c > sharpe_ppo) & (sharpe_a2c > sharpe_ddpg):
            model_ensemble = model_a2c
            model_use.append('A2C')
        else:
            model_ensemble = model_ddpg
            model_use.append('DDPG')

        print(" - 交易从: ", unique_trade_date[i - rebalance_window], "到 ", unique_trade_date[i])
        print("-" * 50)
        last_state_ensemble = DRL_prediction(df=df, model=model_ensemble, name="ensemble",
                                             last_state=last_state_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalance_window,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial)
        
    end = time.time()
    print("集成策略运行时间: ", (end - start) / 60, " 分钟")

# 运行集成策略
run_ensemble_strategy(df=df, 
                      unique_trade_date= unique_trade_date,
                      rebalance_window = rebalance_window,
                      validation_window=validation_window)
