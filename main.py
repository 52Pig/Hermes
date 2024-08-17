# coding=utf8

import random
import configparser
from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount

def main():
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



if __name__ == '__main__':
    main()