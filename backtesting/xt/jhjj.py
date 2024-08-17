# coding:gbk

'''
集合竞价选股回测模型示例（非实盘交易策略）

本策略通过获取沪深300的成份股数据并统计其30天内
开盘价大于前收盘价的天数,并在该天数大于阈值10的时候加入股票池
随后对不在股票池的股票卖出，并买入在股票池不在持仓里的股票'''
# 在指数（例如HS300）日线下运行
import numpy as np


def init(ContextInfo):
    # context.count_bench累计天数阙值
    ContextInfo.count_bench = 10
    # 用于对比的天数
    ContextInfo.count = 30
    # 设置股票池
    s = ContextInfo.get_stock_list_in_sector('沪深300')
    ContextInfo.set_universe(s)
    ContextInfo.accountid = "testS"


def handlebar(ContextInfo):
    buy_sum = 0
    sell_sum = 0
    index = ContextInfo.barpos
    realtimetag = ContextInfo.get_bar_timetag(index)
    print(timetag_to_datetime(realtimetag, '%Y%m%d %H:%M:%S'))
    ContextInfo.holdings = get_holdings(ContextInfo.accountid, "STOCK")
    universe = ContextInfo.get_universe()
    dict_close = ContextInfo.get_history_data(ContextInfo.count, '1d', 'close', 1)
    dict_open = ContextInfo.get_history_data(ContextInfo.count, '1d', 'open', 1)

    trade_symbols = []
    for stock in universe:
        if stock in list(dict_close.keys()):
            open = dict_open[stock]
            pre_close = dict_close[stock]
            try:
                diff = np.array(open[1:]) - np.array(pre_close[:-1])
            except ValueError:
                print('value error:', stock)
            # print 'diff',diff
            # 获取累计天数超过阙值的标的池.并剔除当天没有交易的股票
            if len(diff[diff > 0]) >= ContextInfo.count_bench:
                trade_symbols.append(stock)
        # print '本次股票池有股票数目: ', len(trade_symbols)
    # 如标的池有仓位,平不在标的池的仓位
    for stock_hold, percent_hold in list(ContextInfo.holdings.items()):
        if stock_hold not in trade_symbols:
            order_shares(stock_hold, float(-ContextInfo.holdings[stock_hold]), 'latest', ContextInfo,
                         ContextInfo.accountid)
            sell_sum += 1
        # print '卖出：',stock_hold
    for symbol in trade_symbols:
        if symbol not in ContextInfo.holdings:
            order_shares(symbol, 100, 'latest', ContextInfo, ContextInfo.accountid)
            buy_sum += 1
        # print '买入：',symbol
    if not ContextInfo.do_back_test:
        ContextInfo.paint('buy_num', buy_sum, -1, 0)
        ContextInfo.paint('sell_num', sell_sum, -1, 0)


def get_holdings(accountid, datatype):
    holdinglist = {}
    resultlist = get_trade_detail_data(accountid, datatype, "POSITION")
    for obj in resultlist:
        holdinglist[obj.m_strInstrumentID + "." + obj.m_strExchangeID] = obj.m_nVolume
    return holdinglist