# coding:gbk
'''
ָ����ǿ�ز�ģ��ʾ������ʵ�̽��ײ��ԣ�

��������0.8Ϊ��ʼȨ�ظ���ָ����Ļ���300��Ȩ�ش���0.35%�ĳɷݹ�.
������ռ�İٷֱ�Ϊ(0.8*�ɷݹ�Ȩ��)*100%.Ȼ����ݸ����Ƿ�:
1.��������5�� 2.�����µ�5��
���ж������Ƿ�Ϊǿ�ƹ�/���ƹ�,�������Ȩ����0.8����1.0��0.6'''
# ��ָ��������HS300������������
import numpy as np


def init(ContextInfo):
    # ���ù�Ʊ��
    stock300 = ContextInfo.get_stock_list_in_sector('����300')
    ContextInfo.stock300_weight = {}
    stock300_symbol = []
    stock300_weightlist = []
    ContextInfo.index_code = ContextInfo.stockcode + "." + ContextInfo.market
    for key in stock300:
        # ����Ȩ�ش���0.35%�ĳɷݹ�

        if (ContextInfo.get_weight_in_index(ContextInfo.index_code, key) / 100) > 0.0035:
            stock300_symbol.append(key)
            ContextInfo.stock300_weight[key] = ContextInfo.get_weight_in_index(ContextInfo.index_code, key) / 100
            stock300_weightlist.append(ContextInfo.get_weight_in_index(ContextInfo.index_code, key) / 100)
    print('ѡ��ĳɷֹ�Ȩ���ܺ�Ϊ: ', np.sum(stock300_weightlist))
    ContextInfo.set_universe(stock300_symbol)

    # print ContextInfo.stock300_weight
    # �ʲ����õĳ�ʼȨ��,���Ϊ0.6-0.8-1.0
    ContextInfo.ratio = 0.8

    # �˺�
    ContextInfo.accountid = "testS"


def handlebar(ContextInfo):
    buy_sum = 0
    sell_sum = 0
    index = ContextInfo.barpos
    realtimetag = ContextInfo.get_bar_timetag(index)
    print(timetag_to_datetime(realtimetag, '%Y%m%d %H:%M:%S'))
    dict_close = ContextInfo.get_history_data(7, '1d', 'close', 3)
    # �ֲ���ֵ
    holdvalue = 0
    # �ֲ�
    holdings = get_holdings(ContextInfo.accountid, "STOCK")
    # ʣ���ʽ�
    surpluscapital = get_avaliablecost(ContextInfo.accountid, "STOCK")
    for stock in ContextInfo.stock300_weight:
        if stock in holdings:
            if len(dict_close[stock]) == 7:
                holdvalue += dict_close[stock][-2] * holdings[stock]

    for stock in ContextInfo.stock300_weight:
        # ��û�в�λ���ճ�ʼȨ�ؿ���
        if stock not in holdings and stock in list(dict_close.keys()):
            if len(dict_close[stock]) == 7:
                pre_close = dict_close[stock][-1]
                buy_num = int(ContextInfo.stock300_weight[stock] * (
                            holdvalue + surpluscapital) * ContextInfo.ratio / pre_close / 100)
                order_shares(stock, buy_num * 100, 'fix', pre_close, ContextInfo, ContextInfo.accountid)
                buy_sum += 1
            # print "����",stock,buy_num
        elif stock in list(dict_close.keys()):
            if len(dict_close[stock]) == 7:
                diff = np.array(dict_close[stock][1:6]) - np.array(dict_close[stock][:-2])
                pre_close = dict_close[stock][-1]
                buytarget_num = int(ContextInfo.stock300_weight[stock] * (holdvalue + surpluscapital) * (
                            ContextInfo.ratio + 0.2) / pre_close / 100)
                selltarget_num = int(ContextInfo.stock300_weight[stock] * (holdvalue + surpluscapital) * (
                            ContextInfo.ratio - 0.2) / pre_close / 100)
                # ��ȡ��ȥ5��ļ۸�����,������������Ϊǿ�ƹ�,���ֵ���Ȩ��+0.2���Ĳ�λ
                if all(diff > 0) and holdings[stock] < buytarget_num:
                    buy_num = buytarget_num - holdings[stock]
                    order_shares(stock, buy_num * 100, 'fix', pre_close, ContextInfo, ContextInfo.accountid)
                    buy_sum += 1
                # print "����",stock,buy_num
                # ��ȡ��ȥ5��ļ۸�����,�������µ���Ϊ���ƹ�,���ֵ���Ȩ��-0.2���Ĳ�λ
                elif all(diff < 0) and holdings[stock] > selltarget_num:
                    sell_num = holdings[stock] - selltarget_num
                    order_shares(stock, (-1.0) * sell_num * 100, 'fix', pre_close, ContextInfo, ContextInfo.accountid)
                    sell_sum += 1
                # print "����",stock,sell_num
    if not ContextInfo.do_back_test:
        ContextInfo.paint('buy_num', buy_sum, -1, 0)
        ContextInfo.paint('sell_num', sell_sum, -1, 0)


def get_holdings(accountid, datatype):
    holdinglist = {}
    resultlist = get_trade_detail_data(accountid, datatype, "POSITION")
    for obj in resultlist:
        holdinglist[obj.m_strInstrumentID + "." + obj.m_strExchangeID] = obj.m_nVolume / 100
    return holdinglist


def get_avaliablecost(accountid, datatype):
    result = 0
    resultlist = get_trade_detail_data(accountid, datatype, "ACCOUNT")
    for obj in resultlist:
        result = obj.m_dAvailable
    return result