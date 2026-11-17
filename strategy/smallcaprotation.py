# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import datetime
import time
from pathlib import Path

# 添加项目根目录到sys.path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
sys.path.insert(0, str(project_root))

from xtquant import xtconstant
from xtquant import xtdata
from base_strategy import BaseStrategy

# 配置日志
current_date = datetime.datetime.now().strftime('%Y-%m-%d')
log_dir = "D:/tool/dataset/logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"small_cap_rotation_{current_date}.log")
logger = logging.getLogger("SmallCapRotation")
logger.setLevel(logging.INFO)

# 检查是否已存在FileHandler，避免重复添加
file_handler_exists = any(
    isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(log_file)
    for handler in logger.handlers
)

if not file_handler_exists:
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

logger.info("小市值轮动策略启动")


class SmallCapRotationStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.is_test = "0"

        # 策略参数
        self.rotation_hours = 4  # 调仓周期(小时)
        self.num_stocks = 20  # 持仓股票数量
        self.max_position_per_stock = 0.05  # 单只股票最大仓位比例
        self.min_market_cap = 0  # 最小市值限制(亿元)
        self.max_market_cap = 50  # 最大市值限制(亿元)

        # 状态变量
        self.last_rotation_time = None
        self.stock_pool = []
        self.current_holdings = {}

        # 加载股票池
        self._load_stock_pool()

        logger.info("小市值轮动策略初始化完成")

    def _load_stock_pool(self):
        """加载股票池"""
        try:
            # 获取沪深A股所有股票
            all_stocks = xtdata.get_stock_list_in_sector("沪深A股")

            # 过滤ST股票和停牌股票
            self.stock_pool = [
                stock for stock in all_stocks
                if not stock.startswith(('ST', '*ST')) and
                   not xtdata.is_suspended(stock)
            ]

            logger.info(f"股票池加载完成，共{len(self.stock_pool)}只股票")
        except Exception as e:
            logger.error(f"加载股票池失败: {e}")
            # 备用股票池
            self.stock_pool = ['000001.SZ', '000002.SZ', '000004.SZ', '000005.SZ', '000006.SZ',
                               '600000.SH', '600004.SH', '600006.SH', '600007.SH', '600008.SH']

    def get_market_cap(self, stock_code):
        """获取股票市值(单位:亿元)"""
        try:
            # 获取总股本和最新价格
            total_share = xtdata.get_financial_data(stock_code, "total_share", "latest")
            latest_price = xtdata.get_market_data(stock_code, "close", "1d", "1m", True)[stock_code].iloc[-1]

            if total_share is not None and latest_price is not None:
                # 计算总市值 (总股本*股价，转换为亿元)
                market_cap = total_share * latest_price / 100000000
                return market_cap
        except Exception as e:
            logger.error(f"获取{stock_code}市值失败: {e}")
            return None

        return None

    def get_stock_metrics(self, stock_list):
        """批量获取股票指标"""
        metrics = {}

        try:
            # 获取市值数据
            for stock in stock_list:
                market_cap = self.get_market_cap(stock)
                if market_cap is not None and self.min_market_cap <= market_cap <= self.max_market_cap:
                    metrics[stock] = {
                        'market_cap': market_cap,
                        'pe_ratio': xtdata.get_financial_data(stock, "pe_ratio", "latest"),
                        'pb_ratio': xtdata.get_financial_data(stock, "pb_ratio", "latest"),
                        'turnover_rate': xtdata.get_market_data(stock, "turnover_rate", "1d", "1m", True)[stock].iloc[
                            -1]
                    }
        except Exception as e:
            logger.error(f"获取股票指标失败: {e}")

        return metrics

    def should_rotate(self):
        """判断是否应该调仓"""
        current_time = datetime.datetime.now()

        # 首次运行
        if self.last_rotation_time is None:
            return True

        # 检查时间间隔
        time_diff = (current_time - self.last_rotation_time).total_seconds() / 3600
        return time_diff >= self.rotation_hours

    def select_small_cap_stocks(self, metrics_dict, num_stocks):
        """选择小市值股票"""
        # 按市值排序
        sorted_stocks = sorted(
            metrics_dict.items(),
            key=lambda x: x[1]['market_cap']
        )

        # 选择市值最小的股票
        selected_stocks = [
            stock for stock, metrics in sorted_stocks[:num_stocks]
            if metrics['pe_ratio'] > 0 and  # 排除PE为负的股票
               metrics['pb_ratio'] > 0 and  # 排除PB为负的股票
               metrics['turnover_rate'] > 0.5  # 换手率大于0.5%
        ]

        return selected_stocks

    def calculate_position_size(self, total_asset, stock_prices):
        """计算每只股票的仓位大小"""
        position_sizes = {}
        total_weight = 0

        # 等权重分配
        for stock in stock_prices:
            position_sizes[stock] = 1.0 / len(stock_prices)
            total_weight += position_sizes[stock]

        # 调整权重，确保总和为1
        if total_weight > 0:
            for stock in position_sizes:
                position_sizes[stock] /= total_weight
                # 应用单只股票最大仓位限制
                position_sizes[stock] = min(position_sizes[stock], self.max_position_per_stock)

        return position_sizes

    def execute_rotation(self, xt_trader, acc, account_name):
        """执行调仓操作"""
        logger.info("开始执行调仓")

        try:
            # 获取账户资产信息
            asset_info = xt_trader.query_stock_asset(acc)
            if not asset_info:
                logger.error("获取账户资产信息失败")
                return False

            total_asset = asset_info.total_asset
            logger.info(f"账户总资产: {total_asset:.2f}")

            # 获取当前持仓
            positions = xt_trader.query_stock_positions(acc)
            current_holdings = {
                pos.stock_code: {
                    'volume': pos.volume,
                    'market_value': pos.market_value
                } for pos in positions if pos.volume > 0
            }

            # 获取股票指标
            stock_metrics = self.get_stock_metrics(self.stock_pool)
            if not stock_metrics:
                logger.error("获取股票指标失败")
                return False

            # 选择小市值股票
            selected_stocks = self.select_small_cap_stocks(stock_metrics, self.num_stocks)
            logger.info(f"选中的小市值股票: {selected_stocks}")

            # 获取选中股票的最新价格
            price_data = xtdata.get_market_data(selected_stocks, "close", "1d", "1m", True)
            stock_prices = {
                stock: price_data[stock].iloc[-1]
                for stock in selected_stocks
                if not pd.isna(price_data[stock].iloc[-1])
            }

            # 计算仓位大小
            position_sizes = self.calculate_position_size(total_asset, stock_prices)

            # 卖出不在新组合中的股票
            for stock, holding in current_holdings.items():
                if stock not in selected_stocks:
                    # 执行卖出
                    order_id = xt_trader.order_stock(
                        acc, stock, xtconstant.STOCK_SELL, holding['volume'],
                        xtconstant.LATEST_PRICE, 0, strategy_name="small_cap_rotation"
                    )
                    logger.info(f"卖出 {stock}, 数量 {holding['volume']}, 订单号 {order_id}")

            # 买入新组合中的股票
            for stock in selected_stocks:
                if stock not in stock_prices:
                    continue

                # 计算目标市值
                target_value = total_asset * position_sizes.get(stock, 0)
                current_value = current_holdings.get(stock, {}).get('market_value', 0)

                # 计算需要买入的金额
                buy_value = max(0, target_value - current_value)

                if buy_value > 0:
                    # 计算买入数量
                    price = stock_prices[stock]
                    volume = int(buy_value / price / 100) * 100  # 整手交易

                    if volume > 0:
                        # 执行买入
                        order_id = xt_trader.order_stock(
                            acc, stock, xtconstant.STOCK_BUY, volume,
                            xtconstant.LATEST_PRICE, 0, strategy_name="small_cap_rotation"
                        )
                        logger.info(f"买入 {stock}, 数量 {volume}, 价格 {price:.2f}, 订单号 {order_id}")

            # 更新调仓时间
            self.last_rotation_time = datetime.datetime.now()
            logger.info("调仓完成")
            return True

        except Exception as e:
            logger.error(f"调仓执行失败: {e}")
            return False

    async def do(self, accounts):
        """主执行函数"""
        logger.info(f"执行小市值轮动策略，时间: {datetime.datetime.now()}")

        # 检查是否应该调仓
        if not self.should_rotate():
            logger.info("未到达调仓时间，跳过本次执行")
            return json.dumps({"status": "skipped", "reason": "not rotation time"})

        # 获取账户信息
        req_dict = accounts.get("acc_2", {})
        xt_trader = req_dict.get("xt_trader")
        acc = req_dict.get("account")
        is_test = req_dict.get("is_test", "0")
        self.is_test = is_test
        account_name = req_dict.get("acc_name", "未知")

        if not xt_trader or not acc:
            logger.error("无法获取交易接口或账户信息")
            return json.dumps({"status": "error", "reason": "no trader or account"})

        # 执行调仓
        success = self.execute_rotation(xt_trader, acc, account_name)

        if success:
            return json.dumps({"status": "success", "message": "rotation completed"})
        else:
            return json.dumps({"status": "error", "reason": "rotation failed"})


# 主函数
if __name__ == "__main__":
    # 加载配置
    config = {
        "rotation_hours": 4,
        "num_stocks": 20,
        "max_position_per_stock": 0.05,
        "min_market_cap": 0,
        "max_market_cap": 50
    }

    # 创建策略实例
    strategy = SmallCapRotationStrategy(config)

    # 模拟账户信息
    accounts = {
        "acc_2": {
            "xt_trader": None,  # 实际使用时需要替换为真实的xt_trader对象
            "account": "您的账户ID",
            "is_test": "0",
            "acc_name": "测试账户"
        }
    }

    # 运行策略
    import asyncio

    asyncio.run(strategy.do(accounts))
