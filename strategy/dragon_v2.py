#coding=gbk

import time
import json
import datetime
import a_trade_calendar
from mpmath import limit
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


class Dragon_V2(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.sell_price_history = {}
        self.buy_price_history = {}

    def get_buy_volume(self, current_price, limit_up_days):
        """���ݵ�ǰ�ɼۺ�������ȷ���������"""
        if limit_up_days < 5:
            if 0 < current_price <= 5.0:
                return 600
            elif 5.0 < current_price <= 8.0:
                return 500
            elif 8.0 < current_price <= 10.0:
                return 400
            elif current_price > 10.0:
                return 300
        else:
            return 100
        return 0

    def should_sell(self, stock, last_1d_close_price, max_price, max_price_timestamp, last_price):
        """�ж��Ƿ�������������"""
        # �������1��������max_price>0.095�󳬹�5���ӹɼ۲���>0.095������
        if max_price_timestamp is not None and max_price is not None:
            limit_up_price = round(last_1d_close_price * 1.095, 2)
            current_time = datetime.datetime.now()
            time_diff = (current_time - max_price_timestamp).total_seconds()
            if time_diff > 300 and max_price >= limit_up_price > last_price:
                return True
        # �������2��������5�ιɼ��Ƿ�ȫ������ǰһ�����̼۵�2%
        if stock not in self.sell_price_history or len(self.sell_price_history[stock]) < 5:
            return False  # ���ݲ���

        his_price_list = self.sell_price_history[stock]
        lt_target_num = 0
        if len(his_price_list) > 5:
            his_price_list = his_price_list[:5]
        for price in his_price_list:
            if ( price - last_1d_close_price ) / last_1d_close_price < 0.02:
                lt_target_num += 1
        if lt_target_num == 5:
            return True
        else:
            return False

    def should_buy(self, stock_code, current_price, last_1d_close_price):
        ## ���̼۱��������̼۵���4%��������
        if (current_price - last_1d_close_price) / last_1d_close_price < 0.04:
            return False
        ## ��¼�����5�ε���ʷ�۸�����
        his_price_list = self.buy_price_history[stock_code]
        if len(his_price_list) > 5:
            his_price_list = his_price_list[:5]
        # �۸������жϣ��������µ�����������
        if any(price > current_price for price in his_price_list):
            return False
        # ���������������жϣ�ȷ����Ʊ�۸������ǵ������ԣ����������Ʋ���ʱ����
        if len([price for price in his_price_list if price < current_price]) < 3:
            return False
        return True

    def price_update_callback(self, data, xt_trader, acc, pools_list):
        '''
          1,��ѯ��λ�����в�λ���ж��Ƿ�Ҫ����
          2,��λС��6֧�������̽������
        :param data: �ص�����
        :param xt_trader:
        :param acc:
        :return:
        '''
        #### �����ж�
        ## �鿴�Ƿ�ֲ֣����ֲ����عɼ��Ƿ����Ԥ�ڣ�������Ԥ��������������һֱ����
        ## ��û�гֲ֣����عɼۣ�ѡ������
        # ��¼������־
        sell_list = list()
        # ��ѯ�˻�ί��
        ## ��ǰ�Ƿ���ί�е�,�����ظ����ϵ�
        stock_wt_map = dict()
        wt_infos = xt_trader.query_stock_orders(acc, True)
        for wt_info in wt_infos:
            # print(wt_info.stock_code, wt_info.order_volume, wt_info.price)
            if wt_info.stock_code is not None:
                stock_wt_map[wt_info.stock_code] = 1

        # ��ѯ�ֲֹ�Ʊ
        has_stock_obj = xt_trader.query_stock_positions(acc)
        has_stock_list = list()
        for has_stock in has_stock_obj:
            # print('�ֲ�����ֵ=market_value=', has_stock.market_value)
            # print('�ɱ�=open_price=', has_stock.open_price)
            has_volume = has_stock.volume
            has_stock_code = has_stock.stock_code
            if stock_wt_map.get(has_stock_code, 0) == 1:
                continue
            has_stock_list.append(has_stock_code)
            if has_stock_code not in data:
                # print(f"[ERROR]has_stock_code not in data,has_stock_code={has_stock_code}")
                continue
            last_price = round(data[has_stock_code]['lastPrice'], 2)
            last_1d_close_price = round(data[has_stock_code]['lastClose'], 2)
            max_price = round(data[has_stock_code]['high'], 2)
            max_price_timestamp = None
            # ��¼��߼�ʱ���
            if last_price == max_price and max_price_timestamp is None:
                max_price_timestamp = datetime.datetime.now()
            # last_1d_close_price = utils.get_close_price(has_stock_code, last_n=1)
            # last_price = utils.get_latest_price(has_stock_code, is_download=True)
            print(f"[DEBUG]sell_before,has_volume={has_volume}, last_price={last_price}, last_1d_close_price={last_1d_close_price}, has_stock_code={has_stock_code}")

            # �������5�εĹɼ�����
            if has_stock_code not in self.sell_price_history:
                self.sell_price_history[has_stock_code] = list()
            self.sell_price_history[has_stock_code].append(last_price)
            if len(self.sell_price_history[has_stock_code]) > 5:
                self.sell_price_history[has_stock_code].pop(0)

            # if has_volume > 0 and (last_price - last_1d_close_price) / last_1d_close_price < 0.02:
            if has_volume > 0 and self.should_sell(has_stock_code, last_1d_close_price, max_price, max_price_timestamp, last_price):
                # Ϊ�˱����޷����ӣ��۸��������ƣ������۸��ܵ��ڵ�ǰ�۸��98%
                sell_price = round(last_price * 0.99, 2)
                if sell_price < round(last_1d_close_price - last_1d_close_price * 0.1, 2):
                    sell_price = round(last_1d_close_price - last_1d_close_price * 0.1, 2)
                order_id = xt_trader.order_stock(acc, has_stock_code, xtconstant.STOCK_SELL, has_volume,
                                                 xtconstant.FIX_PRICE, sell_price)
                sell = dict()
                sell['code'] = has_stock_code
                sell['price'] = sell_price
                sell['action'] = 'sell'
                sell['order_id'] = order_id
                sell['volume'] = has_volume
                sell_list.append(sell)

        ## ����ί��
        buy_list = list()
        has_stock_num = len(set(has_stock_list))
        if has_stock_num < 6:
            # ��ѯ�˻����
            acc_info = xt_trader.query_stock_asset(acc)
            cash = acc_info.cash
            for stock_code, limit_up_days, yesterday_volume in pools_list:
                # ## ���Ͼ���ʱ�䣺���ù�Ʊ�ķⵥ���Ƿ���ͬ����������ߣ���������ȡ��ί�С�
                # if jj_start_time <= cur_time <= jj_end_time:
                #     ## ���ù�Ʊ�ķⵥ���Ƿ���ͬ�����������
                #     is_highest = is_highest_bid(stock_code)
                #     if not is_highest:
                #         action_name = 'cancel'
                #         ## ȡ������ί�еĶ���
                #         xt_trader.cancel_order_stock(acc, order_id)
                ## �Ѿ��ֲ֣����ٿ�������
                if stock_code in has_stock_list:
                    print("[DEBUG]buy has_stock_code=", stock_code)
                    continue
                ## ��ǰ�Ƿ���ί��
                if stock_wt_map.get(stock_code, 0) == 1:
                    continue
                ## ��ǰ�۸��������̼۸�
                # current_price = utils.get_latest_price(stock_code, True)
                # last_1d_close_price = utils.get_close_price(stock_code, last_n=1)
                ## û�е�ǰtick����
                if stock_code not in data:
                    print(f"[ERROR]buy stock_code not in data,stock_code={stock_code}")
                    continue
                current_price = data[stock_code]['lastPrice']
                last_1d_close_price = data[stock_code]['lastClose']
                if current_price is None or last_1d_close_price is None:
                    continue
                if current_price <= 0 or last_1d_close_price <= 0:
                    continue

                # �������5�εĹɼ�����
                if stock_code not in self.buy_price_history:
                    self.buy_price_history[stock_code] = list()
                self.buy_price_history[stock_code].append(current_price)
                if len(self.buy_price_history[stock_code]) > 5:
                    self.buy_price_history[stock_code].pop(0)

                ## �Ƿ�������������
                ##   �˻�����㹻���룺�˻����> ��Ʊ�۸�*100
                ##   ��ǰί�е��в����ڴ˹�
                is_buy = self.should_buy(stock_code, current_price, last_1d_close_price)
                if not is_buy:
                    continue
                ## �����λ:
                ##   ���ɼ�<5.0,�������������400��
                ##   ��8.0>=�ɼ�>5.0�����������300��
                ##   ��10.0>=�ɼ�>8.0,�����������200;
                ##   ���ɼ�>10.0,�����������100��
                buy_volume = self.get_buy_volume(current_price, limit_up_days)
                if buy_volume is None or buy_volume == 0:
                    continue
                ## �˻�����㹻����
                if cash >= current_price * buy_volume:
                    order_id = xt_trader.order_stock(acc, stock_code, xtconstant.STOCK_BUY, buy_volume,
                                                     xtconstant.FIX_PRICE, current_price)

                    ret = dict()
                    ret['code'] = stock_code
                    ret['price'] = current_price
                    ret['action'] = 'buy'
                    ret['volume'] = buy_volume
                    ret['order_id'] = order_id
                    buy_list.append(ret)
        ret_list = buy_list + sell_list
        return json.dumps({"msg": ret_list})

    def do(self, accounts):
        print("[DEBUG]do dragon_v2 ", utils.get_current_time(), accounts)
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
            if instrument_detail is None:
                continue
            if instrument_detail is not None:
                stock_name = instrument_detail.get("InstrumentName", "")
                if "ST" in stock_name:
                    # print("[DEBUG]filter_ST=", stock_code, stock_name)
                    continue
            # �ų���ҵ��
            if stock_code.startswith("3"):
                # print("[DEBUG]filter_chuangyeban=", stock_code, stock_name)
                continue
            ## �Ƿ�ͣ��
            ins_status = instrument_detail.get("InstrumentStatus",0)
            if ins_status >= 1:
                print("[DEBUG]=instrumentStatus", ins_status, stock_code, stock_name)
                continue
            recall_stock.append((stock_code, stock_name))
        #print(len(recall_stock), recall_stock)

        ## �ɼۼ۸�ɸѡ
        # xtdata.download_history_data(stock_code, '1m', '20240601')
        slist = [code for code,name in recall_stock]
        print("download start! recall_stock_size=", len(slist))
        xtdata.download_history_data2(slist, period='1d', start_time='20240901')
        print("download finish!")

        eff_stock_list = list()
        for stock_code, stock_name in recall_stock:
            latest_price = utils.get_close_price(stock_code, last_n=1)
            if latest_price is None:
                print("latest_price=", latest_price, stock_code)
                continue
            if latest_price < 2.0 or latest_price > 40.0:
                # print("[DEBUG]filter latest price", stock_code, stock_name)
                continue
            eff_stock_list.append(stock_code)

        # �������N�����������������
        N = 10
        last_n = utils.get_past_trade_date(N)
        #print(f'[DEBUG]last_n={last_n} ;eff_stock_list={len(eff_stock_list)};eff_stock_list={eff_stock_list}')
        pre_ztgp_stocks = get_ztgp_days(eff_stock_list, last_n)
        print(f'[DEBUG]pre_ztgp_stocks={pre_ztgp_stocks}')
        sorted_stocks = filter_and_sort_stocks(pre_ztgp_stocks)
        print(f"[DEBUG]sorted_stocks={sorted_stocks}")
        if len(sorted_stocks) == 0:
            return json.dumps({"msg":[{"mark":"sorted_stocks is empty."}]})

        # ��ͬ�����ɽ���������Ϊ����
        pools_list = list()
        eff_stock_list = list()
        limit_2_index = 0
        limit_3_index = 0
        limit_4_index = 0
        limit_5_index = 0
        for content in sorted_stocks:
            stock_code, limit_up_days, yesterday_volume = content
            if limit_up_days == 2 and limit_2_index == 0:
                pools_list.append(content)
                eff_stock_list.append(stock_code)
                limit_2_index += 1
            elif limit_up_days == 3 and limit_3_index == 0:
                pools_list.append(content)
                eff_stock_list.append(stock_code)
                limit_3_index += 1
            elif limit_up_days == 4 and limit_4_index == 0:
                pools_list.append(content)
                eff_stock_list.append(stock_code)
                limit_4_index += 1
            elif limit_up_days == 5 and limit_5_index == 0:
                pools_list.append(content)
                eff_stock_list.append(stock_code)
                limit_5_index += 1

        ## ������Ч�Ľ����
        if len(pools_list) == 0:
            return json.dumps({"msg":[{"mark":"pools_list is empty."}]})
        print(f"[DEBUG]pools_list={pools_list}")

        ## ����ʱ��
        cur_time = datetime.datetime.now().time()
        gy_time = datetime.datetime.strptime("22:01", "%H:%M").time()
        jj_start_time = datetime.datetime.strptime("09:15", "%H:%M").time()
        jj_end_time = datetime.datetime.strptime("09:19", "%H:%M").time()
        start_time = datetime.datetime.strptime("09:31", "%H:%M").time()
        mid_start_time = datetime.datetime.strptime("11:30", "%H:%M").time()
        mid_end_time = datetime.datetime.strptime("13:01", "%H:%M").time()
        end_time = datetime.datetime.strptime("14:55", "%H:%M").time()
        is_trade_time = start_time <= cur_time <= mid_start_time or mid_end_time <= cur_time <= end_time
        is_jj_time = jj_start_time <= cur_time <= jj_end_time

        ## ���ڽ���ʱ�䲻����
        if not is_trade_time:
            return json.dumps({"msg":[{"mark":"is_not_trade_time."}]})

        # �Ѿ��ֲֹ�ƱҲ��Ҫ�ŵ�������
        has_stock_obj = xt_trader.query_stock_positions(acc)
        has_stock_list = list()
        has_stock_map = dict()
        for has_stock in has_stock_obj:
            # print('�ֲ�����ֵ=market_value=', has_stock.market_value)
            # print('�ɱ�=open_price=', has_stock.open_price)
            has_volume = has_stock.volume
            has_stock_code = has_stock.stock_code
            has_stock_map[has_stock_code] = has_volume
            has_stock_list.append(has_stock_code)
        subscribe_whole_list = list(set(has_stock_list + eff_stock_list))
        print(f"[DEBUG]subscribe_whole_list={subscribe_whole_list}")
        # ע��ȫ�ƻص�����
        # ������һ���յ��б����洢���ؽ��
        final_ret_list = []

        # ע��ȫ�ƻص�����
        def callback(data):
            ret = self.price_update_callback(data, xt_trader, acc, pools_list)
            if ret is not None:
                final_ret_list.extend(ret)

        xtdata.subscribe_whole_quote(subscribe_whole_list, callback=callback)
        xtdata.run()
        # �������Ϻ�Ľ��
        return json.dumps({"msg": final_ret_list})


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

            # �����ص�����
            if stock_code not in volume_data or len(volume_data[stock_code]) < 2:
                continue  # ���ݲ��㣬����

            stock_volume_data = volume_data[stock_code]
            yesterday_volume = stock_volume_data['volume'].iloc[-1]  # ���ճɽ���
            filtered_stocks.append((stock_code, limit_up_days, yesterday_volume))

    # �������ճɽ����Ӵ�С����
    sorted_stocks = sorted(filtered_stocks, key=lambda x: int(x[2]), reverse=True)
    return sorted_stocks

def get_ztgp_days(index_stocks, last_n):
    """��ȡ������ͣ��Ʊ������ͣ����
    Args:
        index_stocks: ��Ʊ�����б�
        start_date: ��ʼ�������ڣ���ʽΪ'YYYYMMDD'
    Returns:
        ��ͣ��Ʊ������ͣ�������б�
    """
    ztgp_stocks = []
    start_date = last_n.replace('-', '')
    # print("----start_date------", start_date)
    for stock_code in index_stocks:
        data = xtdata.get_market_data_ex(
            stock_list=[stock_code],
            field_list=['time', 'close'],
            period='1d',
            start_time=start_date
        )

        # �����ص�����
        if stock_code not in data or len(data[stock_code]) < 2:
            ztgp_stocks.append((stock_code, 0))  # ���ݲ��㣬����0����
            continue

        stock_data = data[stock_code]

        # ��������Ƿ���ͣ
        if (stock_data['close'].iloc[-1] - stock_data['close'].iloc[-2]) / stock_data['close'].iloc[-2] < 0.097:
            ztgp_stocks.append((stock_code, 0))  # ����û����ͣ������0����
            continue

        limit_up_count = 0
        for i in range(len(stock_data) - 1, 0, -1):  # ������
            if (stock_data['close'].iloc[i] - stock_data['close'].iloc[i - 1]) / stock_data['close'].iloc[
                i - 1] >= 0.097:
                limit_up_count += 1
            else:
                break  # ������������ͣ������ֹͣ����

        ztgp_stocks.append((stock_code, limit_up_count))

    return ztgp_stocks