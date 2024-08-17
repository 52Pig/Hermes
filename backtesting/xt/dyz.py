# coding:gbk
"""
多因子选股回测模型示例（非实盘交易策略）
#HS300日线下运行，20个交易日进行 一次调仓，每次买入在买入备选中因子评分前10的股票，每支股票各分配当前可用资金的10%（权重可调整）
#扩展数据需要在补完HS300成分股数据之后生成，本模型中扩展数据暂时使用VBA指标ATR和ADTM生成，命名为atr和adtm
"""
import pandas as pd
import numpy as np
import time
import datetime


def init(ContextInfo):
    ContextInfo.s = ContextInfo.get_sector('000300.SH')
    ContextInfo.set_universe(ContextInfo.s)
    ContextInfo.day = 0
    ContextInfo.holdings = {i: 0 for i in ContextInfo.s}
    ContextInfo.weight = [0.1] * 10  # 设置资金分配权重
    ContextInfo.buypoint = {}
    ContextInfo.money = ContextInfo.capital
    ContextInfo.profit = 0
    ContextInfo.accountID = 'testS'


def handlebar(ContextInfo):
    rank1 = {}
    rank2 = {}
    rank_total = {}
    tmp_stock = {}
    d = ContextInfo.barpos
    price = ContextInfo.get_history_data(1, '1d', 'open', 3)
    if d > 60 and d % 20 == 0:  # 每月一调仓
        nowDate = timetag_to_datetime(ContextInfo.get_bar_timetag(d), '%Y%m%d')
        print(nowDate)
        buys, sells = signal(ContextInfo)
        order = {}
        for k in list(buys.keys()):
            if buys[k] == 1:
                rank1[k] = ext_data_rank('atr', k[-2:] + k[0:6], 0, ContextInfo)
                rank2[k] = ext_data_rank('adtm', k[-2:] + k[0:6], 0, ContextInfo)
                # print rank1[k], rank2[k]
                rank_total[k] = 1.0 * rank1[k]  # 因子的权重需要人为设置，此处取了0.5和-0.5
                print(1111111, rank1[k])
        tmp = sorted(list(rank_total.items()), key=lambda item: item[1])
        # print tmp
        if len(tmp) >= 10:
            tmp_stock = {i[0] for i in tmp[:10]}
        else:
            tmp_stock = {i[0] for i in tmp}  # 买入备选中若超过10只股票则选10支，不足10支则全选
        for k in list(buys.keys()):
            if k not in tmp_stock:
                buys[k] = 0
        if tmp_stock:
            print('stock pool:', tmp_stock)
            for k in ContextInfo.s:
                if ContextInfo.holdings[k] > 0 and sells[k] == 1:
                    print('ready to sell')
                    order_shares(k, -ContextInfo.holdings[k] * 100, 'fix', price[k][-1], ContextInfo,
                                 ContextInfo.accountID)
                    ContextInfo.money += price[k][-1] * ContextInfo.holdings[k] * 100 - 0.0003 * ContextInfo.holdings[
                        k] * 100 * price[k][-1]  # 手续费按万三设定
                    ContextInfo.profit += (price[k][-1] - ContextInfo.buypoint[k]) * ContextInfo.holdings[
                        k] * 100 - 0.0003 * ContextInfo.holdings[k] * 100 * price[k][-1]
                    # print price[k][-1]
                    print(k)
                    # print ContextInfo.money
                    ContextInfo.holdings[k] = 0
            ContextInfo.money_distribution = {k: i * ContextInfo.money for (k, i) in zip(tmp_stock, ContextInfo.weight)}
            for k in tmp_stock:
                if ContextInfo.holdings[k] == 0 and buys[k] == 1:
                    print('ready to buy')
                    order[k] = int(ContextInfo.money_distribution[k] / (price[k][-1])) / 100
                    order_shares(k, order[k] * 100, 'fix', price[k][-1], ContextInfo, ContextInfo.accountID)
                    ContextInfo.buypoint[k] = price[k][-1]
                    ContextInfo.money -= price[k][-1] * order[k] * 100 - 0.0003 * order[k] * 100 * price[k][-1]
                    ContextInfo.profit -= 0.0003 * order[k] * 100 * price[k][-1]
                    print(k)
                    ContextInfo.holdings[k] = order[k]
            print(ContextInfo.money, ContextInfo.profit, ContextInfo.capital)
    profit = ContextInfo.profit / ContextInfo.capital
    if not ContextInfo.do_back_test:
        ContextInfo.paint('profit_ratio', profit, -1, 0)


def signal(ContextInfo):
    buy = {i: 0 for i in ContextInfo.s}
    sell = {i: 0 for i in ContextInfo.s}
    data_high = ContextInfo.get_history_data(22, '1d', 'high', 3)
    data_high_pre = ContextInfo.get_history_data(2, '1d', 'high', 3)
    data_close60 = ContextInfo.get_history_data(62, '1d', 'close', 3)
    # print data_high
    # print data_close
    # print data_close60
    for k in ContextInfo.s:
        if k in data_close60:
            if len(data_high_pre[k]) == 2 and len(data_high[k]) == 22 and len(data_close60[k]) == 62:
                if data_high_pre[k][-2] > max(data_high[k][:-2]):
                    buy[k] = 1  # 超过20日最高价，加入买入备选
                elif data_high_pre[k][-2] < np.mean(data_close60[k][:-2]):
                    sell[k] = 1  # 低于60日均线，加入卖出备选
    # print buy
    # print sell
    return buy, sell  # 买入卖出备选