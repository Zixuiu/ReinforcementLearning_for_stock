# 导入random，用于生成随机数  
import random  
  
# 导入gym，一个提供开发和比较强化学习算法的开源库  
import gym  
  
# 导入numpy，一个用于数值计算和数据处理的Python库  
import numpy as np  
  
# 导入spaces，这是gym库的一部分，用于创建具有特定空间类型的action_space和observation_space  
from gym import spaces  
  
# 导入yaml，用于读取YAML配置文件  
import yaml  
  
# 打开并读取名为'config.yaml'的配置文件，将其中内容存储到args变量中  
with open('config.yaml') as f:  
    args = yaml.safe_load(f)  
  
# 定义一些最大值，这些值在后面的代码中用于缩放或限制一些值，例如账户余额、股票数量、股票价格、交易量等。  
MAX_ACCOUNT_BALANCE = 2147483647  
MAX_NUM_SHARES = 2147483647  
MAX_SHARE_PRICE = 5000  
MAX_VOLUME = 1000e8  
MAX_AMOUNT = 3e10  
MAX_OPEN_POSITIONS = 5  
MAX_STEPS = 20000  
MAX_DAY_CHANGE = 1  
  
# 从配置文件中获取初始账户余额  
INITIAL_ACCOUNT_BALANCE = args['env_args']['initial_account_balance']  
  
# 定义一个函数，用于将数据框中的某些列进行特征工程处理，然后返回这些处理后的特征值  
def feature_engineer(df, current_step):  
    # 选择指定的列，并除以相应的最大值进行归一化处理  
    feas = [  
        df.loc[current_step, 'open'] / MAX_SHARE_PRICE,  
        df.loc[current_step, 'high'] / MAX_SHARE_PRICE,  
        df.loc[current_step, 'low'] / MAX_SHARE_PRICE,  
        df.loc[current_step, 'close'] / MAX_SHARE_PRICE,  
        df.loc[current_step, 'volume'] / MAX_VOLUME,  
    ]  
    # 返回处理后的特征值列表  
    return feas  
  
# 定义一个名为StockTradingEnv的类，继承自gym.Env，这是一个OpenAI gym环境的基础类  
class StockTradingEnv(gym.Env):  
    """A stock trading environment for OpenAI gym"""  
    # 设置元数据，这里指定了渲染模式为'human'  
    metadata = {'render.modes': ['human']}  
  
    # 初始化函数，接收一个数据框参数df  
    def __init__(self, df):  
        # 调用父类的初始化函数  
        super(StockTradingEnv, self).__init__()  
        # 存储传入的df参数  
        self.df = df  
        # 定义奖励的范围，这里使用的是账户余额的范围，表示最大奖励和最小奖励都应该在这个范围内。下面的代码中也多次使用了这个范围。  
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)  
        # 定义动作空间，这里的动作格式是Buy x%, Sell x%, Hold等，且是一个二维的Box空间，取值范围在[0,3]和[0,1]之间。这个Box空间的定义和下面的observation space一起，构成了强化学习Agent可以感知和操作的环境。  
        self.action_space = spaces.Box(  
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)  
        # 定义观测空间，观测空间的定义和上面的动作空间类似，也是一个二维的Box空间，取值范围在[0,1]之间。这个空间表示Agent在环境中能获取到的信息或状态。在后面的代码中，通过调用_next_observation函数，可以得到下一个时间步的观测值。  
        self.observation_space = spaces.Box(  
            low=0, high=1, shape=(11,), dtype=np.float16)  
  
    # 定义一个名为_next_observation的函数，该函数是类的一部分，它没有明确的输入参数，但是使用了类的一些属性。  
    def _next_observation(self):  
      
        # 调用名为feature_engineer的函数，传入两个参数，分别是self.df和self.current_step。这个函数似乎是用来处理或生成特征的。  
        # 返回的结果被存储在feas变量中。  
        feas = feature_engineer(self.df, self.current_step)  
      
        # 创建一个名为obs的numpy数组，它由feas和其他几个元素组成。  
        # 这些元素是：self.balance除以MAX_ACCOUNT_BALANCE，self.max_net_worth除以MAX_ACCOUNT_BALANCE，  
        # self.shares_held除以MAX_NUM_SHARES，self.cost_basis除以MAX_SHARE_PRICE，  
        # self.total_shares_sold除以MAX_NUM_SHARES，self.total_sales_value除以(MAX_NUM_SHARES * MAX_SHARE_PRICE)。  
        obs = np.array(feas + [  
            self.balance / MAX_ACCOUNT_BALANCE,  
            self.max_net_worth / MAX_ACCOUNT_BALANCE,  
            self.shares_held / MAX_NUM_SHARES,  
            self.cost_basis / MAX_SHARE_PRICE,  
            self.total_shares_sold / MAX_NUM_SHARES,  
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),  
        ])  
      
        # 返回生成的观察值obs。  
        return obs
    
    def _take_action(self, action):  
        # 设置当前价格为一个在当前时间步内的随机价格，范围在开盘价和收盘价之间  
        # 获取当前时间步的开盘价和收盘价，然后使用random.uniform函数生成一个介于两者之间的随机数作为当前价格  
        current_price = random.uniform(  
            self.df.loc[self.current_step, "open"], self.df.loc[self.current_step, "close"])  
      
        # 获取action中的第一个元素作为行动类型，第二个元素作为数量  
        action_type = action[0]  
        amount = action[1]  
      
        # 如果行动类型小于1，表示购买操作  
        if action_type < 1:  
            # 计算需要购买的数量，这里假设购买金额为当前余额的百分之amount，所以额外需要花费的金额为余额*amount  
            additional_cost = self.balance * amount  
            # 如果额外需要花费的金额小于等于20000元，则按照额外需要花费的金额计算购买的股票数量  
            if additional_cost <= 20000:  
                shares_bought = (additional_cost-5)/current_price # 此时手续费为5元  
            else:  
                # 如果额外需要花费的金额大于20000元，则按照更高的购买价格计算购买的股票数量，这里的购买价格为当前价格*(1+交易佣金费率)  
                # 买入价格为 current_price * (1+交易佣金费率)  
                shares_bought = additional_cost / (current_price * (1 + args['env_args']['commission']))  
            # 保存之前的成本价和持有的股票数  
            prev_cost = self.cost_basis * self.shares_held  
            # 减少当前余额  
            self.balance -= additional_cost  
            # 计算新的成本价，公式为之前成本价乘以持有的股票数加上额外花费的金额，再除以新的持有股票数  
            self.cost_basis = (  
                prev_cost + additional_cost) / (self.shares_held + shares_bought)  
            # 增加持有的股票数  
            self.shares_held += shares_bought  
      
        # 如果行动类型小于2，表示卖出操作  
        elif action_type < 2:  
            # 计算需要卖出的股票数量，这里假设卖出份额为持有的股票数的百分之amount  
            shares_sold = int(self.shares_held * amount)  
            # 计算卖出获得的现金，公式为卖出的股票数乘以当前股票价格再乘以(1-交易佣金费率)  
            # 卖出获得的现金为: 卖出份额*当前股票价格*(1-交易佣金费率)  
            current_get_balance = shares_sold * current_price * (1 - args['env_args']['commission'])  
            # 增加当前余额  
            self.balance += current_get_balance  
            # 减少持有的股票数  
            self.shares_held -= shares_sold  
            # 增加卖出的总份额  
            self.total_shares_sold += shares_sold  
            # 增加卖出的总价值，公式为卖出的现金加上卖出的股票数乘以当前股票价格再乘以(1-交易佣金费率)  
            self.total_sales_value += current_get_balance  
      
        # 计算当前的总资产净值，公式为当前余额加上持有的股票数乘以当前股票价格再乘以(1-交易佣金费率)  
        # 当前持有总资产:现金+股票  
        self.net_worth = self.balance + self.shares_held * current_price * (1 - args['env_args']['commission'])  
      
        # 如果当前的总资产净值大于最大总资产净值，则更新最大总资产净值  
        if self.net_worth > self.max_net_worth:  
            self.max_net_worth = self.net_worth  
      
        # 如果持有的股票数为0，则将成本价设为0  
        if self.shares_held == 0:  
            self.cost_basis = 0

      # 定义一个名为step的函数，该函数接受一个参数action  
    def step(self, action):  
        # 在环境中执行一次步骤  
        # 调用私有方法_take_action，传入action作为参数  
        self._take_action(action)  
      
        # 初始化变量done为False，表示此次步骤还未完成  
        done = False  
      
        # 将当前步骤数增加1  
        self.current_step += 1  
      
        # 判断当前步骤数是否大于df中'open'列的值数减1  
        if self.current_step > len(self.df.loc[:, 'open'].values) - 1:  
            # 如果超出范围，将当前步骤数设置为0，进行循环训练  
            # self.current_step = 0  # loop training  
            self.current_step = random.randint(6, len(self.df) - 1)  
            # 设置done为True，表示此次步骤已经完成  
            # done = True  
      
        # reward 1:  
        # 这部分的代码计算第一种奖励机制，如果net_worth大于初始账户余额INITIAL_ACCOUNT_BALANCE，则奖励为1，否则为-100  
        # reward = self.net_worth - INITIAL_ACCOUNT_BALANCE  
        # reward = 1 if reward > 0 else -100  
        # 需要修改 self.reward_range = (0, MAX_ACCOUNT_BALANCE)  
      
        # reward 2:  
        # 这部分的代码计算第二种奖励机制，根据当前步骤数与最大步骤数的比例来计算奖励值  
        delay_modifier = (self.current_step / MAX_STEPS)  
        reward = self.balance * delay_modifier  
      
        # 如果net_worth小于等于0，设置done为True，表示此次步骤已经完成  
        if self.net_worth <= 0:  
            done = True  
      
        # 调用私有方法_next_observation，生成下一个观察值  
        obs = self._next_observation()  
      
        # 返回观察值、奖励值、是否完成以及一个空的字典作为元数据  
        return obs, reward, done, {}

       # 定义一个名为reset的函数，它是环境对象的一个方法  
    def reset(self, new_df=None, test=False):  
        # 将环境的初始状态设置为初始账户余额  
        self.balance = INITIAL_ACCOUNT_BALANCE  
        # 将环境的初始净值设置为初始账户余额  
        self.net_worth = INITIAL_ACCOUNT_BALANCE  
        # 将环境的最大净值设置为初始账户余额  
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE  
        # 初始化持有的股票数为0  
        self.shares_held = 0  
        # 初始化成本基数为0  
        self.cost_basis = 0  
        # 初始化卖出的总份额为0  
        self.total_shares_sold = 0  
        # 初始化卖出的总价值为0  
        self.total_sales_value = 0  
      
        # 如果提供了新的数据集，将其传递给环境  
        if new_df:  
            self.df = new_df  
      
        # 将当前步骤设置为数据帧内的随机点  
        self.current_step = 0  
      
        # 调用环境对象的_next_observation方法并返回结果  
        return self._next_observation()  
      
    # 定义一个名为render的函数，它是环境对象的一个方法  
    def render(self, mode='human', close=False):  
        # 计算当前净值与初始账户余额的差值，即利润  
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE  
        # 获取当前步骤的日期  
        date = self.df.loc[self.current_step - 1, "date"]  
        # 获取当前步骤的开盘价  
        open = self.df.loc[self.current_step - 1, "open"]  
        # 获取当前步骤的收盘价  
        close = self.df.loc[self.current_step - 1, "close"]  
        # 获取当前步骤的高价  
        high = self.df.loc[self.current_step - 1, "high"]  
        # 获取当前步骤的低价  
        low = self.df.loc[self.current_step - 1, "low"]  
      
        # 打印分隔线，用于分隔输出信息  
        print('-'*30)  
        # 打印日期信息  
        print(f'date: {date}')  
        # 打印当前步骤信息  
        print(f'Step: {self.current_step}')  
        # 打印当前余额信息  
        print(f'Balance: {self.balance}')  
        # 打印持有的股票数以及卖出的总份额信息  
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')  
        # 打印持有股票的平均成本以及总销售额信息  
        print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')  
        # 打印当前净值以及最大净值信息  
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')  
        # 打印利润信息  
        print(f'Profit: {profit}')  
      
        # 打印开盘价、收盘价、高价和低价信息  
        print(f'open:{open} close:{close}, high:{high}, low: {low}')  
      
        # 返回日期、利润、开盘价、收盘价、高价和低价信息  
        return date, profit, open, close, high, low
