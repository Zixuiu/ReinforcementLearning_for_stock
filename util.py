import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from stable_baselines3.common.vec_env import DummyVecEnv
from env.SingleStockEnv import StockTradingEnv
import pandas as pd

def prepare_env(stock_file):
    # 导入股票数据
    df = pd.read_csv(stock_file)
    # 排序数据
    df = df.sort_values('date')
    # 用 DummyVecEnv 封装交易环境
    env = DummyVecEnv([lambda: StockTradingEnv(df)])
    return env, len(df)

def find_file(path, name):
    # 查找指定文件名的文件路径
    for root, dirs, files in os.walk(path):
        for fname in files:
            if name in fname:
                return os.path.join(root, fname)

def plot_daily_profits(stock_code, RL_model, daily_profits, dates, daily_opens, daily_closes, daily_highs, daily_lows):
    # 创建子图布局
    fig = make_subplots(rows=2, cols=1, subplot_titles=("profit", "daily price"), shared_xaxes=True)

    # 添加盈利曲线
    fig.add_trace(
        go.Scatter(x=dates, y=daily_profits, mode='lines+markers'),
        row=1, col=1
    )
    # 添加K线图（按日期绘制开盘、收盘、最高、最低价格）
    fig.add_trace(
        go.Candlestick(x=dates, open=daily_opens, high=daily_highs, low=daily_lows, close=daily_closes),
        row=2, col=1
    )
    # 修改布局
    fig.update_layout(xaxis_rangeslider_visible=False, showlegend=False, title_text=f"{stock_code}, {RL_model}")
    # 显示图形
    fig.show()

    # 创建存储图像的文件夹
    #os.makedirs('./img/', exist_ok=True)
    # 将图像保存为png格式
  #  fig.write_image(f'./img/{stock_code + "_" + RL_model}.png')
