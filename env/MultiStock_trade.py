import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

# 每次交易的股票数量的归一化因子
# 每次交易100股
HMAX_NORMALIZE = 100
# 账户中的初始资金
INITIAL_ACCOUNT_BALANCE = 1000000
# 投资组合中股票的总数
STOCK_DIM = 30
# 交易费用：收益的千分之一的合理金额
TRANSACTION_FEE_PERCENT = 0.001

# 紊乱指标：90-150是合理的阈值
#TURBULENCE_THRESHOLD = 140
REWARD_SCALING = 1e-4

class StockEnvTrade(gym.Env):
    """用于OpenAI gym的股票交易环境"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day=0, turbulence_threshold=140, initial=True, previous_state=[], model_name='', iteration=''):
        self.day = day
        self.df = df
        self.initial = initial
        self.previous_state = previous_state
        # 动作空间从-1到1进行归一化
        # 形状为(STOCK_DIM,)
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))
        # 观察空间的形状为181：[当前余额] + [价位 1-30] + [拥有股票 1-30]
        # + [高 1-30] + [低 1-30] + [收盘价 1-30] + [成交量 1-30]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(181,))
        # 从Pandas DataFrame中加载数据
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # 初始化状态
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     self.data.open.values.tolist() + \
                     [0]*STOCK_DIM + \
                     self.data.high.values.tolist() + \
                     self.data.low.values.tolist() + \
                     self.data.close.values.tolist() + \
                     self.data.volume.values.tolist()
        # 初始化奖励
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        # 用于存储所有总资产变化的记录
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self._seed()
        self.model_name = model_name
        self.iteration=iteration


    def _sell_stock(self, index, action):
        # 根据动作的符号进行卖出操作
        if self.turbulence < self.turbulence_threshold:
            if self.state[index+STOCK_DIM+1] > 0:
                # 更新账户余额
                self.state[0] += self.state[index+1] * min(abs(action), self.state[index+STOCK_DIM+1]) * (1 - TRANSACTION_FEE_PERCENT)

                self.state[index+STOCK_DIM+1] -= min(abs(action), self.state[index+STOCK_DIM+1])
                self.cost += self.state[index+1] * min(abs(action), self.state[index+STOCK_DIM+1]) * TRANSACTION_FEE_PERCENT
                self.trades += 1
            else:
                pass
        else:
            # 如果紊乱指数超过阈值，清空所有持仓
            if self.state[index+STOCK_DIM+1] > 0:
                # 更新账户余额
                self.state[0] += self.state[index+1] * self.state[index+STOCK_DIM+1] * (1 - TRANSACTION_FEE_PERCENT)
                self.state[index+STOCK_DIM+1] = 0
                self.cost += self.state[index+1] * self.state[index+STOCK_DIM+1] * TRANSACTION_FEE_PERCENT
                self.trades += 1
            else:
                pass
    
    def _buy_stock(self, index, action):
        # 根据动作的符号进行购买操作
        if self.turbulence < self.turbulence_threshold:
            available_amount = self.state[0] // self.state[index + 1]
            
            # 更新账户余额
            self.state[0] -= self.state[index + 1] * min(available_amount, action) * (1 + TRANSACTION_FEE_PERCENT)

            self.state[index + STOCK_DIM + 1] += min(available_amount, action)
            
            self.cost += self.state[index + 1] * min(available_amount, action) * TRANSACTION_FEE_PERCENT
            self.trades += 1
        else:
            # 如果紊乱指数超过阈值，停止购买
            pass
            
    def step(self, actions):
        # 判断当前时间步是否为最后一个时间步
        self.terminal = self.day >= len(self.df.index.unique()) - 1
    
        if self.terminal:
            # 当时间步为最后一个时间步时，保存模型的账户价值，并打印一些相关信息
            plt.plot(self.asset_memory,'r')
            plt.savefig('/Users/poteman/learn/RL/ReinforcementLearning_for_stock/archive/account_value_trade_{}_{}.png'.format(self.model_name, self.iteration))
            plt.close()
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('/Users/poteman/learn/RL/ReinforcementLearning_for_stock/archive/account_value_trade_{}_{}.csv'.format(
                self.model_name, self.iteration))
            end_total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM+1)]) * np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
    
            print("previous_total_asset:{}".format(self.asset_memory[0]))
            print("end_total_asset:{}".format(end_total_asset))
            print("total_reward:{}".format(self.state[0] + sum(np.array(self.state[1:(STOCK_DIM+1)]) * np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))- self.asset_memory[0] ))
            print("total_cost: ", self.cost)
            print("total trades: ", self.trades)
    
            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
            print("Sharpe: ",sharpe)
    
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv('/Users/poteman/learn/RL/ReinforcementLearning_for_stock/archive/account_rewards_trade_{}_{}.csv'.format(self.model_name, self.iteration))
    
            return self.state, self.reward, self.terminal, {}
    
        else:
            # 对行动进行标准化
            actions = actions * HMAX_NORMALIZE
    
            if self.turbulence >= self.turbulence_threshold:
                actions = np.array([-HMAX_NORMALIZE] * STOCK_DIM)
    
            # 计算当前资产总价值
            begin_total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM+1)]) * np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
    
            # 根据行动进行买卖股票
            argsort_actions = np.argsort(actions)
    
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]
    
            for index in sell_index:
                self._sell_stock(index, actions[index])
    
            for index in buy_index:
                self._buy_stock(index, actions[index])
    
            # 进入下一个时间步
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.turbulence = self.data['turbulence'].values[0]
    
            # 更新状态
            self.state = [self.state[0]] + \
                    self.data.open.values.tolist() + \
                    list(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]) + \
                    self.data.high.values.tolist() + \
                    self.data.low.values.tolist() + \
                    self.data.close.values.tolist() + \
                    self.data.volume.values.tolist()
    
            # 计算当前账户总价值和回报
            end_total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM+1)]) * np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            self.asset_memory.append(end_total_asset)
    
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
    
            # 根据回报进行调整
            self.reward = self.reward * REWARD_SCALING
    
        return self.state, self.reward, self.terminal, {}

