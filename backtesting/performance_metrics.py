import numpy as np
import pandas as pd

"""
包含各种绩效指标的计算函数
"""

def calculate_max_drawdown(portfolio_values):
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    return np.min(drawdown)

def calculate_volatility(portfolio_values):
    return np.std(portfolio_values) / np.mean(portfolio_values)

def calculate_calmar_ratio(portfolio_values, max_drawdown):
    annual_return = (portfolio_values[-1] / portfolio_values[0]) ** (252 / len(portfolio_values)) - 1
    return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

def calculate_sortino_ratio(portfolio_values):
    returns = pd.Series(portfolio_values).pct_change().dropna()
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns)
    expected_return = np.mean(returns)
    return expected_return / downside_std if downside_std != 0 else 0
