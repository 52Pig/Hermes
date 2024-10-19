import os
import sys
import logging
import pandas as pd
import concurrent.futures
from datetime import datetime
import numpy as np
from backtesting.performance_metrics import calculate_max_drawdown, calculate_volatility, calculate_calmar_ratio, \
    calculate_sortino_ratio

if not os.path.exists("logs"):
    os.makedirs('logs')

# 设置日志
current_date = datetime.now().strftime('%Y-%m-%d')
logging.basicConfig(filename=f'logs/backtest_{current_date}.log', level=logging.INFO, format='%(asctime)s %(message)s')


class BacktestEngine:
    def __init__(self, initial_capital, data, strategies):
        self.initial_capital = initial_capital
        self.data = data
        self.strategies = strategies
        self.strategy_results = {strategy.name: self._init_strategy_result(strategy.name) for strategy in strategies}

    def _init_strategy_result(self, strategy_name):
        return {
            'cash': self.initial_capital,
            'positions': 0,
            'portfolio_value': self.initial_capital,
            'trade_log': [],
            'portfolio_values': [],
            'max_drawdown': 0,
            'volatility': 0,
            'calmar_ratio': 0,
            'sortino_ratio': 0
        }

    def run_backtest(self):
        logging.info("开始回测")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._run_strategy, strategy): strategy.name for strategy in self.strategies}
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                    logging.info(f"{name} 策略回测完成")
                except Exception as e:
                    logging.error(f"{name} 策略回测失败: {e}")

    def _run_strategy(self, strategy):
        result = self.strategy_results[strategy.name]
        for index, row in self.data.iterrows():
            signal = strategy.generate_signal(row)
            if signal == 'buy' and result['cash'] > 0:
                result['positions'] += result['cash'] / row['Close']
                result['cash'] = 0
                result['trade_log'].append({'Date': row['Date'], 'Type': 'buy', 'Price': row['Close']})
            elif signal == 'sell' and result['positions'] > 0:
                result['cash'] += result['positions'] * row['Close']
                result['positions'] = 0
                result['trade_log'].append({'Date': row['Date'], 'Type': 'sell', 'Price': row['Close']})

            result['portfolio_value'] = result['cash'] + result['positions'] * row['Close']
            result['portfolio_values'].append(result['portfolio_value'])

        # 计算绩效指标
        self._calculate_performance(result)

    def _calculate_performance(self, result):
        result['max_drawdown'] = calculate_max_drawdown(result['portfolio_values'])
        result['volatility'] = calculate_volatility(result['portfolio_values'])
        result['calmar_ratio'] = calculate_calmar_ratio(result['portfolio_values'], result['max_drawdown'])
        result['sortino_ratio'] = calculate_sortino_ratio(result['portfolio_values'])

    def save_logs(self):
        logging.info("保存回测日志")
        for strategy_name, result in self.strategy_results.items():
            pd.DataFrame(result['trade_log']).to_csv(f'{strategy_name}_trade_log.csv')