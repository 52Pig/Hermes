# coding:gbk
"""
������ѡ�ɻز�ģ��ʾ������ʵ�̽��ײ��ԣ�
#HS300���������У�20�������ս��� һ�ε��֣�ÿ�����������뱸ѡ����������ǰ10�Ĺ�Ʊ��ÿ֧��Ʊ�����䵱ǰ�����ʽ��10%��Ȩ�ؿɵ�����
#��չ������Ҫ�ڲ���HS300�ɷֹ�����֮�����ɣ���ģ������չ������ʱʹ��VBAָ��ATR��ADTM���ɣ�����Ϊatr��adtm
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
    ContextInfo.weight = [0.1] * 10  # �����ʽ����Ȩ��
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
    if d > 60 and d % 20 == 0:  # ÿ��һ����
        nowDate = timetag_to_datetime(ContextInfo.get_bar_timetag(d), '%Y%m%d')
        print(nowDate)
        buys, sells = signal(ContextInfo)
        order = {}
        for k in list(buys.keys()):
            if buys[k] == 1:
                rank1[k] = ext_data_rank('atr', k[-2:] + k[0:6], 0, ContextInfo)
                rank2[k] = ext_data_rank('adtm', k[-2:] + k[0:6], 0, ContextInfo)
                # print rank1[k], rank2[k]
                rank_total[k] = 1.0 * rank1[k]  # ���ӵ�Ȩ����Ҫ��Ϊ���ã��˴�ȡ��0.5��-0.5
                print(1111111, rank1[k])
        tmp = sorted(list(rank_total.items()), key=lambda item: item[1])
        # print tmp
        if len(tmp) >= 10:
            tmp_stock = {i[0] for i in tmp[:10]}
        else:
            tmp_stock = {i[0] for i in tmp}  # ���뱸ѡ��������10ֻ��Ʊ��ѡ10֧������10֧��ȫѡ
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
                        k] * 100 * price[k][-1]  # �����Ѱ������趨
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
                    buy[k] = 1  # ����20����߼ۣ��������뱸ѡ
                elif data_high_pre[k][-2] < np.mean(data_close60[k][:-2]):
                    sell[k] = 1  # ����60�վ��ߣ�����������ѡ
    # print buy
    # print sell
    return buy, sell  # ����������ѡ