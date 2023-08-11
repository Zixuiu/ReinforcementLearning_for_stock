import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

# shares normalization factor
# 100 shares per trade
HMAX_NORMALIZE = 100
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE=1000000
# total number of stocks in our portfolio
STOCK_DIM = 30
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4

class StockEnvTrain(gym.Env):  
    """A stock trading environment for OpenAI gym"""  
    # 定义这个类是gym.Env的子类，用于创建OpenAI gym的环境。  
    # 这个环境是用于股票交易的。  
  
    metadata = {'render.modes': ['human']}  
    # 设置环境的元数据，这里指定了环境的渲染模式为'human'，也就是人类可读的格式。  
  
    def __init__(self, df,day = 0):  
        # 定义StockEnvTrain类的构造函数  
        # super(StockEnv, self).__init__()  
        # 调用父类的构造函数，初始化一些父类中定义的状态和行为。  
        # money = 10 , scope = 1  
        # 设置初始的账户余额为10，scope未定义，可能是交易的股票范围。  
        self.day = day  
        # 保存输入的DataFrame的索引，也就是要处理的日期。  
        self.df = df  
        # 保存输入的pandas DataFrame。  
  
        # action_space normalization and shape is STOCK_DIM  
        self.action_space = spaces.Box(low = -1, high = 1,shape = (STOCK_DIM,))   
        # 定义动作空间为一个矩形空间，范围是-1到1，形状是STOCK_DIM。  
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30]   
        # +[high 1-30]+ [low 1-30] + [close 1-30] + [volume 1-30]  
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (181,))  
        # 定义观测空间为一个矩形空间，范围是0到无穷大，形状是181。  
        # load data from a pandas dataframe  
        # 从pandas DataFrame中加载数据。  
        self.data = self.df.loc[self.day,:]  
        # 从df中加载指定日期的数据。  
        self.terminal = False               
        # 初始化终端状态标志为False。  
        # initalize state  
        self.state = [INITIAL_ACCOUNT_BALANCE] + \  
                      self.data.open.values.tolist() + \  
                      [0]*STOCK_DIM + \  
                      self.data.high.values.tolist() + \  
                      self.data.low.values.tolist() + \  
                      self.data.close.values.tolist() + \  
                      self.data.volume.values.tolist()  
        # 初始化状态，这里的状态包括了账户余额、开盘价、最高价、最低价、收盘价和交易量等信息。  
        # initialize reward  
        self.reward = 0  
        self.cost = 0  
        # 初始化奖励和成本。  
        # memorize all the total balance change  
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]  
        self.rewards_memory = []  
        self.trades = 0  
        # 记录所有账户余额的变化。  
        # self.reset()  
        # 重置环境，返回到初始状态。这一行被注释掉了，可能是还未实现。  
        self._seed()  
        # 设置随机数种子，保证结果的可重复性。
  def _sell_stock(self, index, action):  
    # 定义一个名为_sell_stock的内嵌函数，该函数接受两个参数：self（表示该函数是一个类的方法）和index（表示要交易的股票的索引）。此外，还有一个名为action的参数，它代表要进行的交易量，其正负分别代表卖出和买入。  
    # 检查个人的股票余额是否大于0，这是卖出股票的前提条件  
    if self.state[index+STOCK_DIM+1] > 0:  
        # 根据卖出或买入的股票数量和股票价格来更新账户余额，同时考虑到手续费  
        #update balance  
        self.state[0] += \  
        self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * \  
         (1- TRANSACTION_FEE_PERCENT)  
  
        # 从股票余额中减去卖出的股票数量  
        self.state[index+STOCK_DIM+1] -= min(abs(action), self.state[index+STOCK_DIM+1])  
        # 累加此次交易的手续费  
        self.cost +=self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * \  
         TRANSACTION_FEE_PERCENT  
        # 累加交易次数  
        self.trades+=1  
    else:  
        pass  # 如果股票余额为0，则不进行任何操作  
  
    def _buy_stock(self, index, action):  
        # 定义一个名为_buy_stock的内嵌函数，该函数接受两个参数：self（表示该函数是一个类的方法）和index（表示要交易的股票的索引）。此外，还有一个名为action的参数，它代表要进行的交易量，其正负分别代表卖出和买入。  
        # 计算可以购买的股票数量，这个数量不能超过账户余额除以股票价格  
        available_amount = self.state[0] // self.state[index+1]  
        # 打印可以购买的股票数量  
        # print('available_amount:{}'.format(available_amount))  
      
        # 根据购买数量和股票价格来更新账户余额，同时考虑到手续费  
        #update balance  
        self.state[0] -= self.state[index+1]*min(available_amount, action)* \  
                          (1+ TRANSACTION_FEE_PERCENT)  
      
        # 在股票持有量中增加购买的股票数量  
        self.state[index+STOCK_DIM+1] += min(available_amount, action)  
      
        # 累加此次交易的手续费  
        self.cost+=self.state[index+1]*min(available_amount, action)* \  
                          TRANSACTION_FEE_PERCENT  
        # 累加交易次数  
        self.trades+=1
    def step(self, actions):  
        # 打印当前天数  
        # print(self.day)  
        self.terminal = self.day >= len(self.df.index.unique())-1  
        # 判断是否达到数据集的最后一天，如果是则标记为终止状态  
        # print(actions)  
      
        if self.terminal:  
            # 在终端时，绘制并保存资产变化图  
            plt.plot(self.asset_memory,'r')  
            plt.savefig('/Users/poteman/learn/RL/ReinforcementLearning_for_stock/archive/account_value_train.png')  
            plt.close()  
            # 计算最终的总资产值，包括持有的股票价值和现金价值  
            end_total_asset = self.state[0]+ \  
                sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))  
      
            # 打印最终的总资产值  
            #print("end_total_asset:{}".format(end_total_asset))  
            # 将资产变化值保存为DataFrame并导出为CSV文件  
            df_total_value = pd.DataFrame(self.asset_memory)  
            df_total_value.to_csv('/Users/poteman/learn/RL/ReinforcementLearning_for_stock/archive/account_value_train.csv')  
            # 打印总奖励值（即总资产值减去初始资产值）  
            #print("total_reward:{}".format(self.state[0]+sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):61]))- INITIAL_ACCOUNT_BALANCE ))  
            # 打印总成本  
            #print("total_cost: ", self.cost)  
            # 打印交易次数  
            #print("total_trades: ", self.trades)  
            df_total_value.columns = ['account_value']  
            df_total_value['daily_return']=df_total_value.pct_change(1)  
            # 计算夏普比率  
            sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ \  
                  df_total_value['daily_return'].std()  
            # 打印夏普比率  
            #print("Sharpe: ",sharpe)  
            # 打印分割线  
            #print("=================================")  
            df_rewards = pd.DataFrame(self.rewards_memory)  
            #df_rewards.to_csv('/kaggle/working/account_rewards_train.csv')  
      
            # 打印总资产值（包括持有的股票价值和现金价值）  
            # print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))  
            # 使用pickle模块将当前状态保存到文件中，以便后续使用  
            #with open('obs.pkl', 'wb') as f:    
            #    pickle.dump(self.state, f)  
      
            return self.state, self.reward, self.terminal,{}

        else:  
            # 这段代码在上述条件不满足时执行  
            # 打印self.state中的第二个到第二十九个元素（数组切片表示）  
            # np.array将结果转换为numpy数组，因为print函数不能直接打印数组  
            # 此行代码未实际执行，因为else块中没有np.array(self.state[1:29])的定义  
            #print(np.array(self.state[1:29]))  
          
            # 将动作归一化（乘以HMAX_NORMALIZE）  
            actions = actions * HMAX_NORMALIZE  
          
            # 改为将动作转换为整数类型（之前可能为浮点数）  
            #actions = (actions.astype(int))  
            actions = actions.astype(int)  
          
            # 计算当前总资产，它等于当前资产加上当前持有的每个股票的价值总和  
            begin_total_asset = self.state[0]+ \  
                                sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))  
                                #print("begin_total_asset:{}".format(begin_total_asset))  
                                # 此行代码未实际执行，因为begin_total_asset的定义没有在else块中使用  
          
            # 按照动作值的大小对所有动作进行排序，返回排序后的索引  
            argsort_actions = np.argsort(actions)  
          
            # 提取所有负值（卖出）的动作索引和所有正值（买入）的动作索引  
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]  
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]  
          
            # 对于卖出动作索引，执行卖出股票的操作  
            for index in sell_index:  
                # print('take sell action'.format(actions[index]))  
                self._sell_stock(index, actions[index])  
          
            # 对于买入动作索引，执行买入股票的操作  
            for index in buy_index:  
                # print('take buy action: {}'.format(actions[index]))  
                self._buy_stock(index, actions[index])  
          
            # 增加天数，即时间步数增加1  
            self.day += 1  
          
            # 加载新的一天（新的一天）的数据  
            self.data = self.df.loc[self.day,:]  
          
            # 加载下一个状态（即新一天的状态）  
            # load next state  
            # print("stock_shares:{}".format(self.state[29:]))  
            self.state =  [self.state[0]] + \  
                        self.data.open.values.tolist() + \  
                        list(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]) + \  
                        self.data.high.values.tolist() + \  
                        self.data.low.values.tolist() + \  
                        self.data.close.values.tolist() + \  
                        self.data.volume.values.tolist()  
          
            # 计算当前总资产，它等于初始资产加上当前持有的每个股票的价值总和  
            end_total_asset = self.state[0]+ \  
                              sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))  
            self.asset_memory.append(end_total_asset)  
            # 计算奖励，即当前总资产与初始资产之差（在强化学习中，奖励通常是与目标相关的）  
            self.reward = end_total_asset - begin_total_asset              
            self.rewards_memory.append(self.reward)  
            # 返回当前状态，奖励，终端标记（这行代码中未定义），和空字典作为额外信息（这行代码中未定义）  
        return self.state, self.reward, self.terminal, {}

    def reset(self):  # 定义一个名为reset的方法，该方法属于该类，因此它可以在该类的其他方法中被调用。  
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]  # 将账户余额初始值存储在asset_memory中，这是一个列表，只有一个元素，即初始账户余额。  
            self.day = 0  # 将day设置为0，表示还未处理任何交易日。  
            self.data = self.df.loc[self.day,:]  # 从DataFrame中获取第0天的数据，并将其存储在data中。  
            self.cost = 0  # 初始化成本为0。  
            self.trades = 0  # 初始化交易次数为0。  
            self.terminal = False  # 标记当前状态为非终止状态，即可以进行更多的交易。  
            self.rewards_memory = []  # 初始化奖励记忆为空列表，用于存储每一步的奖励值。  
            # 初始化状态  
            #initiate state  
            self.state = [INITIAL_ACCOUNT_BALANCE] + \  # 将初始账户余额添加到状态中  
                          self.data.open.values.tolist() + \  # 将open价格转换为列表并添加到状态中  
                          [0]*STOCK_DIM + \  # 添加一些零，数量为STOCK_DIM（可能表示股票数量）  
                          self.data.high.values.tolist() + \  # 将high价格转换为列表并添加到状态中  
                          self.data.low.values.tolist() + \  # 将low价格转换为列表并添加到状态中  
                          self.data.close.values.tolist() + \  # 将close价格转换为列表并添加到状态中  
                          self.data.volume.values.tolist()  # 将交易量转换为列表并添加到状态中  
            # iteration += 1  # 这行代码在这里没有实际作用，因为 reset 方法通常在开始新的训练周期时调用，所以这个增量没有意义。  
            return self.state  # 返回当前状态。  
      
    def render(self, mode='human'):  # 定义一个名为render的方法，该方法返回当前的状态。如果mode='human'，则直接返回，否则可能需要进一步处理。  
            return self.state  # 返回当前状态。  
      
    def _seed(self, seed=None):  # 定义一个名为_seed的私有方法，用于设置随机数生成器的种子。这个方法在测试或重现实验时很有用。  
            self.np_random, seed = seeding.np_random(seed)  # 使用种子设置numpy的随机数生成器。  
            return [seed]  # 返回设置的种子。
