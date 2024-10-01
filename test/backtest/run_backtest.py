import os
import sys
import pandas as pd

# 设置项目根路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root_dir))

from backtesting.backtest_engine import BacktestEngine
from backtesting.strategy import SimpleMovingAverageStrategy
from backtesting.plotting_tool import plot_portfolio_performance
from backtesting.genetic_optimize import optimize_strategy_parameters

def backtest_example():
    # 读取历史数据
    data = pd.read_csv("dataset/historical_data.csv")
    data['short_mavg'] = data['Close'].rolling(window=40, min_periods=1).mean()
    data['long_mavg'] = data['Close'].rolling(window=100, min_periods=1).mean()

    # 初始化策略
    sma_strategy = SimpleMovingAverageStrategy(name="SMA Strategy")

    # 初始化回测引擎
    engine = BacktestEngine(initial_capital=100000, data=data, strategies=[sma_strategy])

    # 运行回测
    engine.run_backtest()

    # 绘制绩效图
    plot_portfolio_performance(data, engine.strategy_results)

    # 优化策略参数
    param_space = [(10, 50), (50, 200)]
    optimize_strategy_parameters(engine, sma_strategy, param_space)

if __name__ == "__main__":
    backtest_example()
