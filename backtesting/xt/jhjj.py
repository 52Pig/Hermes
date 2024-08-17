# coding:gbk

'''
���Ͼ���ѡ�ɻز�ģ��ʾ������ʵ�̽��ײ��ԣ�

������ͨ����ȡ����300�ĳɷݹ����ݲ�ͳ����30����
���̼۴���ǰ���̼۵�����,���ڸ�����������ֵ10��ʱ������Ʊ��
���Բ��ڹ�Ʊ�صĹ�Ʊ�������������ڹ�Ʊ�ز��ڳֲ���Ĺ�Ʊ'''
# ��ָ��������HS300������������
import numpy as np


def init(ContextInfo):
    # context.count_bench�ۼ�������ֵ
    ContextInfo.count_bench = 10
    # ���ڶԱȵ�����
    ContextInfo.count = 30
    # ���ù�Ʊ��
    s = ContextInfo.get_stock_list_in_sector('����300')
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
            # ��ȡ�ۼ�����������ֵ�ı�ĳ�.���޳�����û�н��׵Ĺ�Ʊ
            if len(diff[diff > 0]) >= ContextInfo.count_bench:
                trade_symbols.append(stock)
        # print '���ι�Ʊ���й�Ʊ��Ŀ: ', len(trade_symbols)
    # ���ĳ��в�λ,ƽ���ڱ�ĳصĲ�λ
    for stock_hold, percent_hold in list(ContextInfo.holdings.items()):
        if stock_hold not in trade_symbols:
            order_shares(stock_hold, float(-ContextInfo.holdings[stock_hold]), 'latest', ContextInfo,
                         ContextInfo.accountid)
            sell_sum += 1
        # print '������',stock_hold
    for symbol in trade_symbols:
        if symbol not in ContextInfo.holdings:
            order_shares(symbol, 100, 'latest', ContextInfo, ContextInfo.accountid)
            buy_sum += 1
        # print '���룺',symbol
    if not ContextInfo.do_back_test:
        ContextInfo.paint('buy_num', buy_sum, -1, 0)
        ContextInfo.paint('sell_num', sell_sum, -1, 0)


def get_holdings(accountid, datatype):
    holdinglist = {}
    resultlist = get_trade_detail_data(accountid, datatype, "POSITION")
    for obj in resultlist:
        holdinglist[obj.m_strInstrumentID + "." + obj.m_strExchangeID] = obj.m_nVolume
    return holdinglist