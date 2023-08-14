import yaml
from stable_baselines3 import PPO, A2C, DDPG, TD3
from util import find_file, plot_daily_profits, prepare_env

with open('config.yaml') as f:
    args = yaml.safe_load(f)

# 该代码是用于测试训练好的强化学习模型在股票交易环境中的表现，并绘制每日收益图。

def load_model(RL_model, stock_code):
    # 加载训练好的模型
    if RL_model == 'A2C':
        model = A2C.load(f"./check_points/{RL_model}_{stock_code}")
    elif RL_model == 'PPO':
        model = PPO.load(f"./check_points/{RL_model}_{stock_code}")
    elif RL_model == 'DDPG':
        model = DDPG.load(f"./check_points/{RL_model}_{stock_code}")
    elif RL_model == 'TD3':
        model = TD3.load(f"./check_points/{RL_model}_{stock_code}")

    return model

# 定义一个名为 test_model 的函数，该函数接受四个参数：test_env（测试环境），len_test（测试数据的长度），以及 model（需要被测试的模型）  
def test_model(test_env, len_test, model):  
  
    # 初始化几个空列表，用于存储测试过程中的数据  
    # dates 列表用于存储日期信息  
    # daily_profits 列表用于存储每日的收益  
    # daily_opens 列表用于存储每日的开盘价  
    # daily_closes 列表用于存储每日的收盘价  
    # daily_highs 列表用于存储每日的最高价  
    # daily_lows 列表用于存储每日的最低价  
    dates = []  
    daily_profits = []  
    daily_opens = []  
    daily_closes = []  
    daily_highs = []  
    daily_lows = []  
  
    # 使用 test_env 的 reset 方法重置环境，获取第一次的观测值（obs）  
    obs = test_env.reset()  
  
    # 进入循环，循环次数为 len_test - 1，因为不需要对最后一个数据进行操作和反馈  
    for i in range(len_test - 1):  
        # 使用模型预测在当前观测下的行动（action）以及对应的状态（_states）  
        action, _states = model.predict(obs)  
          
        # 在环境中执行行动，得到新的观测值，以及对应的奖励、是否结束等信息  
        obs, rewards, done, info = test_env.step(action)  
          
        # 从环境中渲染数据，分别得到日期、收益、开盘价、收盘价、最高价和最低价  
        date, profit, open, close, high, low = test_env.render()  
          
        # 将这些数据添加到对应的列表中  
        dates.append(date)  
        daily_profits.append(profit)  
        daily_opens.append(open)  
        daily_closes.append(close)  
        daily_highs.append(high)  
        daily_lows.append(low)  
          
        # 如果当前步骤导致了环境的结束，那么就跳出循环，不再继续后面的步骤  
        if done:  
            break  
  
    # 返回收集到的数据，包括日期、每日收益、开盘价、收盘价、最高价和最低价  
    return dates, daily_profits, daily_opens, daily_closes, daily_highs, daily_lows
def test_strategy(stock_code, RL_model):
    # 获得股票文件路径
    stock_file = find_file('./data/tushare_data/test', str(stock_code))
    # 准备环境
    test_env, len_test = prepare_env(stock_file)

    # 加载训练好的模型
    model = load_model(RL_model, stock_code)

    # 测试模型并绘制日盈利图
    dates, daily_profits, daily_opens, daily_closes, daily_highs, daily_lows = test_model(test_env, len_test, model)
    plot_daily_profits(stock_code, RL_model, daily_profits, dates, daily_opens, daily_closes, daily_highs, daily_lows)

if __name__ == '__main__':
    # 从配置文件中读取参数
    test_strategy(args['train_args']['stock_code'], args['train_args']['rl_model'])
