#coding=gbk

import time
import datetime
import a_trade_calendar
from xtquant import xtdata
from xtquant import xtconstant
from base_strategy import BaseStrategy
from utils import utils
'''
�����߼�

������
1��ǰһ����������ͣ�Ļ���
2���ų�ST����
3���ų�������2Ԫ����,40Ԫ����
4�����̼���x%��������


���룺
1��ÿ���ӹɼ���2%���ϣ������������Ч����������0.01
2�����д����룬������

���ӣ�
1��һ�������������һ֧���������������һ֧��
'''


class Dragon_V1(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)

    def do(self, accounts):
        print("[DEBUG]do dragon_v1 ", utils.get_current_time(), accounts)
        target_code = '����A��'
        req_dict = accounts.get("acc_1", {})
        xt_trader = req_dict.get("xt_trader")
        acc = req_dict.get("account")
        # acc_name = req_dict.get("acc_name")

        # ��ѯ�������й�Ʊ
        index_stocks = xtdata.get_stock_list_in_sector(target_code)
        ## ɸѡ����Ч�ٻس�
        recall_stock = list()
        for stock_code in index_stocks:
            if not stock_code.endswith(".SH") and not stock_code.endswith(".SZ"):
                continue
            ## �ų�ST�Ĺ�Ʊ
            instrument_detail = xtdata.get_instrument_detail(stock_code)
            #if '300615' in stock_code:
            #print('[DEBUG]instrument_detail=',instrument_detail)
            stock_name = ''
            if instrument_detail is not None:
                stock_name = instrument_detail.get("InstrumentName", "")
                if "ST" in stock_name:
                    # print("[DEBUG]filter_ST=", stock_code, stock_name)
                    continue
            # �ų���ҵ��
            if stock_code.startswith("3"):
                # print("[DEBUG]filter_chuangyeban=", stock_code, stock_name)
                continue
            recall_stock.append((stock_code, stock_name))
        #print(len(recall_stock), recall_stock)

        ## �ɼۼ۸�ɸѡ
        # xtdata.download_history_data(stock_code, '1m', '20240601')
        xtdata.download_history_data2(recall_stock, period='1m', start_time='20240601')

        eff_stock_list = list()
        for stock_code, stock_name in recall_stock:
            latest_price = utils.get_latest_price(stock_code)
            if latest_price < 2.0 or latest_price > 40.0:
                print("[DEBUG]filter latest price", stock_code, stock_name)
                continue
            eff_stock_list.append(stock_code)

        # �������N�����������������
        N = 10
        last_n = utils.get_past_trade_date(N)
        pre_ztgp_stocks = get_ztgp_days(eff_stock_list, last_n)
        sorted_stocks = filter_and_sort_stocks(pre_ztgp_stocks)
        if len(sorted_stocks) == 0:
            return dict()
        stock_code, limit_up_days, yesterday_volume = sorted_stocks[0]
        #
        cur_time = utils.get_current_time()
        gy_time = datetime.datetime.strptime("22:01", "%H:%M").time()
        jj_time = datetime.datetime.strptime("09:18", "%H:%M").time()
        start_time = datetime.datetime.strptime("09:31", "%H:%M").time()
        mid_start_time = datetime.datetime.strptime("11:30", "%H:%M").time()
        mid_end_time = datetime.datetime.strptime("13:01", "%H:%M").time()
        end_time = datetime.datetime.strptime("14:55", "%H:%M").time()

        # �Ҹ�ҹ������
        gy_order_id = ''
        if cur_time == gy_time:
            ## ����ί��
            action_name = "buy"
            # ��ѯ�˻����
            acc_info = xt_trader.query_stock_asset(acc)
            cash = acc_info.cash
            # ��ѯ��ǰ�ɼ�
            current_price = utils.get_latest_price(stock_code)
            ## ���˻����> ��Ʊ�۸�*100��������
            # �µ�
            if current_price is not None:
                if cash >= current_price * 100:
                    order_id = xt_trader.order_stock(acc, stock_code, xtconstant.STOCK_BUY, 100,
                                                     xtconstant.FIX_PRICE, current_price)
                    gy_order_id = order_id
        elif cur_time == jj_time:
            ## ���ù�Ʊ�ķⵥ���Ƿ���ͬ�����������
            is_highest = is_highest_bid(stock_code)
            if not is_highest:
                action_name = 'cancel'
                ## ȡ������ί�еĶ���
                xt_trader.cancel_order_stock(acc, gy_order_id)

        elif start_time <= cur_time <= mid_start_time or mid_end_time <= cur_time <= end_time:
            # ��鵱ǰʱ���Ƿ��� 9:30 �� 15:00 ֮��
            #1��ÿ���Ӽ��ɼ����ƣ����ɼ� < ǰһ�����̼���������
            #2��ÿ���Ӽ��ɼ����ƣ����ɼ� > ǰһ�����̼ۣ�����(�ɼ� - ǰһ�����̼�) / ǰһ�����̼� > 5 % ����й۲죬�� < 5 % ���Ե�ǰ�۸�������
            print("action_name")
            current_price = utils.get_latest_price(stock_code)
            yesterday_close_price = utils.get_yesterday_close_price(stock_code)
            if ( current_price - yesterday_close_price ) / yesterday_close_price < 0.02:
                action_name = "sell"
                # ��ѯ�ֲ���ֵ
                acc_info = xt_trader.query_stock_asset(acc)
                marketValue = acc_info.m_dMarketValue
                # ����
                if current_price is not None:
                    if marketValue > 0:
                        order_id = xt_trader.order_stock(acc, stock_code, xtconstant.STOCK_SELL, 100,
                                                         xtconstant.FIX_PRICE, current_price)





def is_highest_bid(self, stock_code):
    """���ù�Ʊ�ķⵥ���Ƿ�����ͬ�����������"""
    # ��ȡ�ù�Ʊ��������
    limit_up_days = self.calculate_limit_up_days(stock_code)

    # ��ȡ��ͬ�����������й�Ʊ����ⵥ��
    same_limit_up_stocks = self.get_same_limit_up_stocks(limit_up_days)

    if not same_limit_up_stocks:
        return False  # û���ҵ���ͬ�������Ĺ�Ʊ

    # ��ȡ�ù�Ʊ�ķⵥ��
    current_bid_volume = self.get_current_bid_volume(stock_code)

    # ���ⵥ���Ƿ�Ϊ���
    is_highest = all(current_bid_volume >= volume for _, volume in same_limit_up_stocks)
    return is_highest


def get_yesterday_date():
    """ ��ȡ��Aǰһ�������յ����� """
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    # print(today)
    previous_trade_date = a_trade_calendar.get_pre_trade_date(today).replace('-', '')
    # print(previous_trade_date)
    #return (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y%m%d')
    return previous_trade_date


def filter_and_sort_stocks(ztgp_stocks):
    """�ų�2�����º�5��������ͣ�Ĺ�Ʊ���������ճɽ�������"""
    filtered_stocks = []
    for stock_code, limit_up_days in ztgp_stocks:
        if 2 <= limit_up_days <= 5:
            # ��ȡ���յĳɽ�������
            volume_data = xtdata.get_market_data_ex(
                stock_list=[stock_code],
                field_list=['time', 'volume'],
                period='1d',
                start_time='20240601'  # ������Ҫ����ʱ�䷶Χ
            )

            if len(volume_data) < 2:
                continue  # ���ݲ��㣬����
            yesterday_volume = volume_data['volume'][-1]  # ���ճɽ���
            filtered_stocks.append((stock_code, limit_up_days, yesterday_volume))

    # �������ճɽ����Ӵ�С����
    sorted_stocks = sorted(filtered_stocks, key=lambda x: x[2], reverse=True)
    return sorted_stocks


def get_ztgp_days(index_stocks, last_n):
    """�����ȡ������ͣ��Ʊ�ĺ���
     �洢��ͣ��Ʊ���б�
    """
    ztgp_stocks = []
    for stock_code in index_stocks:
        data = xtdata.get_market_data_ex(
            stock_list=[stock_code],
            field_list=['time', 'close'],
            period='1d',
            start_time=last_n
        )

        if len(data) < 2:
            ztgp_stocks.append((stock_code, 0))  # ���ݲ��㣬����0����
            continue

        # ��������Ƿ���ͣ
        if (data['close'][1] - data['close'][0]) / data['close'][0] < 0.097:
            ztgp_stocks.append((stock_code, 0))  # ����û����ͣ������0����
            continue

        limit_up_count = 0
        for i in range(len(data) - 1, 0, -1):  # ������
            if (data['close'][i] - data['close'][i - 1]) / data['close'][i - 1] >= 0.097:
                limit_up_count += 1
            else:
                break  # ������������ͣ������ֹͣ����

        ztgp_stocks.append((stock_code, limit_up_count))

    return ztgp_stocks