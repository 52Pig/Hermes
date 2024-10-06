# coding=utf8

import random
import configparser
# from xtquant.xttrader import XtQuantTrader
# from xtquant.xttype import StockAccount

import time
import asyncio
import xtquant
# from xtquant import xtdata, xttrader

async def main():
    # 初始化配置
    config = configparser.ConfigParser()
    config.read('conf/config.ini')
    # config.read(['conf/config.ini'])
    path = config.get("client", 'mini_path')
    acc_name = config.get("client", "acc_name")
    # print('[mini_path]', path)
    session_id = int(random.randint(100000, 999999))
    xt_trader = XtQuantTrader(path, session_id)
    # 链接qmt客户端
    xt_trader.start()
    connect_result = xt_trader.connect()
    # 订阅账户
    acc = StockAccount(acc_name)
    subsribe_result = xt_trader.subscribe(acc)

    print('[DEBUG]connect_status=', connect_result, ',subscribe_status=', subsribe_result)

    ## 初始化策略管理器
    manager = StrategyManager()

    # 假设监控三只股票
    manager.add_strategy('000001.SZ')  # 股票代码1
    manager.add_strategy('000002.SZ')  # 股票代码2
    manager.add_strategy('000003.SZ')  # 股票代码3

    # 启动所有策略
    await asyncio.run(manager.run_all_strategies())



if __name__ == '__main__':
    asyncio.run(main())
