#coding=gbk
import os
import time
import glob
import json
import asyncio
import datetime
import a_trade_calendar
import random
from xtquant import xtdata
from xtquant import xtconstant
from base_strategy import BaseStrategy
from utils import utils

"""
todo:
1, ������ͣ���ܹ�ʽ��
�������� = ����volume / ǰ5��ƽ��volume

��������	�ź�����	��������
<0.5	������	�߳ɹ�����
0.5-1.2	������	�ֳ����
>3.0	������	�����׷���
2, �����ʽ�ʶ��
ɸѡ������
�ɽ��� > 5��Ԫ������С�̹����������壩
�󵥾�����ռ�� > 20%
�ʽ�����ʽ��
��������� = ���󵥽�� + �󵥽��

3�� ϴ��ǿ���ж�
������ͣ������

������� < 5%������һ�ְ���ѣ�

��ͣ����� < 2%�������ȶ���
�����źţ�
if ��� > 15% and ��ͣ���� > 3: ������ذ�

4�� ����ṹ����
����������	�г�����
<5%	��������
5%-15%	��������
>25%	��������
��̬��ֵ���㣺
�������� = ��ͨ�ɱ��л�Ծ������� �� 1.5
"""


class Dragon_V5(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.is_test = False
        self.sell_price_history = {}
        self.buy_price_history = {}
        self.industry_num_map = dict()
        self.sorted_gpzt_pools = list()

        ## ��������ǿ��ָ��
        strength_file = self.get_latest_file("./logs", "v1_daily_limit_strength_data")
        self.strenth_dict = json.load(open(strength_file))

        ## ���غ�ѡ��
        # pools_file = './logs/dragon_v5_data.20241130'
        pools_file = self.get_latest_file('./logs', 'dragon_v5_data')
        self.stock_dict = json.load(open(pools_file))

        self.pools_list = list()
        keys_list = list(self.stock_dict.keys())
        random.shuffle(keys_list)
        for kk in keys_list:
            # print(kk)
            idict = self.stock_dict.get(kk, {})
            cc = idict.get("code", "")
            is_target = idict.get("is_target", "")
            is_up_limit_before_half_year = idict.get("is_up_limit_before_half_year", "0")
            continuous_up_limit_days = idict.get("continuous_up_limit_days", 0)
            if len(cc) > 0 and is_up_limit_before_half_year == "1" and len(self.pools_list)<60 and continuous_up_limit_days > 0:
                self.pools_list.append(kk)
        print('[dragon_v5][INIT]load pools file name:', pools_file)
        print('[INIT]load total size:', len(keys_list), keys_list[:5])
        print('[INIT]load target size:', len(self.pools_list), self.pools_list[:5])
        print('[INIT]SUCCEED!')

    def get_latest_file(self, directory, prefix):
        # ��ȡĿ¼�������� 'prefix' ��ͷ���ļ�
        files = [
            f for f in glob.glob(os.path.join(directory, f"{prefix}*"))
            if f.endswith('.json')  # ��������
        ]
        # ���û���ҵ��ļ������� None
        if not files:
            return None

        # print(files)
        # ���ļ�������ȡ���ڲ��֣����ҵ����µ��ļ�
        def extract_date(file_path):
            # ʾ���ļ�����reverse_moving_average_bull_track_20250303.json
            base_name = os.path.basename(file_path)  # ��ȡ�ļ�������
            print(base_name)
            date_str = base_name.split('_')[-1].split('.')[0]  # �ָ�� YYYYMMDD
            return int(date_str)  # ת��Ϊ�������ڱȽ�

        # �����ڽ��������ȡ�����ļ�
        files_sorted = sorted(files, key=extract_date, reverse=True)
        latest_file = files_sorted[0]
        # latest_file = max(files, key=lambda x: x.split('.')[-1])  # �����ڲ��ֱȽ��ļ���
        print(f"[INIT]latest_file={latest_file}")
        return latest_file

    def get_buy_volume(self, current_price, limit_up_days):
        """���ݵ�ǰ�ɼۺ�������ȷ���������"""
        """
                    ���ݹ�Ʊ�۸��Ŀ���ܽ�����Ӧ����Ĺ�Ʊ�����������ּ��㣬1��=100�ɣ�
                    :param price: ��Ʊ�۸�1.0~30.0Ԫ��
                    :param target_amount: Ŀ���ܽ���10000Ԫ��
                    :return: ��Ʊ���������ٹ�����
                """
        target_amount = 3000
        if current_price <= 0 or target_amount <= 0:
            return 0  # ����Ƿ�����
        # �����������������ܺ�С����
        ideal_hands = target_amount / (current_price * 100)
        # ��ȡ��ѡ�������ذ�ֵ���컨��ֵ��
        floor_hands = int(ideal_hands)
        ceil_hands = floor_hands + 1

        # ��������������ʵ���ܽ��
        amount_floor = floor_hands * current_price * 100
        amount_ceil = ceil_hands * current_price * 100

        # �Ƚ��ĸ����ӽ�Ŀ�������ѡ�񲻳�֧�ķ�����
        diff_floor = abs(target_amount - amount_floor)
        diff_ceil = abs(target_amount - amount_ceil)

        # ��������ȣ�����ѡ�����С�ķ�������9000 vs 11000ʱѡ9000��
        if diff_floor <= diff_ceil:
            return floor_hands * 100
        else:
            return ceil_hands * 100

        # if limit_up_days < 5:
        #     if 0 < current_price <= 5.0:
        #         return 600
        #     elif 5.0 < current_price <= 8.0:
        #         return 500
        #     elif 8.0 < current_price <= 10.0:
        #         return 400
        #     elif current_price > 10.0:
        #         return 300
        # else:
        #     return 100
        # return 0

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
            if ( price - last_1d_close_price ) / last_1d_close_price < 0.04:
                lt_target_num += 1
        if lt_target_num == 5:
            return True
        else:
            return False

    def should_buy(self, stock_code, current_price, last_1d_close_price, limit_up_days):
        ## ��������ǿ��̫��������
        score = self.strenth_dict.get("score", 0.0)
        zha_rate = self.strenth_dict.get("zha_rate", 0.0)
        all_limit_up_num = self.strenth_dict.get("all_limit_up_num", 0.0)
        if score > 0 and zha_rate > 0 and all_limit_up_num > 0:
            if (zha_rate > 0.59 or all_limit_up_num < 30) and score < 1.6:
                return False

        ## ���̼۱��������̼۵���4%��������
        if limit_up_days == 1 and (current_price - last_1d_close_price) / last_1d_close_price < 0.083:
            return False
        elif limit_up_days == 2 and (current_price - last_1d_close_price) / last_1d_close_price < 0.066:
            return False
        elif (current_price - last_1d_close_price) / last_1d_close_price < 0.04:
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

    def price_update_callback(self, data, xt_trader, acc, pools_list, is_jj_time, is_open_time):
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

                code = wt_info.stock_code.split('.')[0]
                industry = self.stock_dict.get(code, {}).get("industry", "")
                if industry in self.industry_num_map:
                    self.industry_num_map[industry] += 1
                else:
                    self.industry_num_map[industry] = 1

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
            # print(f"wt about stock_code={data[has_stock_code]}")

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
            cash = 0
            if acc_info is not None:
                cash = acc_info.cash
            else:
                return json.dumps({"warn": [{"mark": "cash is None."}]})
            # for stock_code, limit_up_days, yesterday_volume, bidVol, askVol, bidPrice, askPrice \
            #         in pools_list:
            for code in pools_list:
                # print('====', code)
                info_dict = self.stock_dict.get(code.split('.')[0], {})
                stock_code = info_dict.get("code", "")
                limit_up_days = info_dict.get("continuous_up_limit_days", 0)
                industry = info_dict.get("industry", "")
                # print(f'-------{code}==={limit_up_days}==={stock_code}==={info_dict}')
                ## �Ѿ��ֲ֣����ٿ�������
                if stock_code in has_stock_list:
                    # print("[DEBUG]buy has_stock_code=", stock_code)
                    continue
                ## ��ǰ�Ƿ���ί��
                if stock_wt_map.get(stock_code, 0) == 1:
                    continue
                ## һ����ҵ���ֲ�һ֧
                if industry in self.industry_num_map:
                    continue

                ## ��ǰ�۸��������̼۸�
                # current_price = utils.get_latest_price(stock_code, True)
                # last_1d_close_price = utils.get_close_price(stock_code, last_n=1)
                ## û�е�ǰtick����
                if stock_code not in data:
                    # print(f"[ERROR]buy stock_code not in data,stock_code={stock_code}")
                    continue

                # print(f"buy about stock_code={data[stock_code]}")
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
                if not is_jj_time and len(self.buy_price_history[stock_code]) > 5:
                    self.buy_price_history[stock_code].pop(0)
                if is_jj_time:
                    ## ����ʱ�䲻ִ������
                    continue
                if is_open_time:
                    ## �������Ʋ�����
                    price_list = self.buy_price_history[stock_code]
                    is_dec = is_price_declining(price_list)
                    if is_dec:
                        continue

                ## �Ƿ�������������
                ##   �˻�����㹻���룺�˻����> ��Ʊ�۸�*100
                ##   ��ǰί�е��в����ڴ˹�
                is_buy = self.should_buy(stock_code, current_price, last_1d_close_price, limit_up_days)
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

    async def do(self, accounts):
        print("[DEBUG]do dragon_v5 ", utils.get_current_time(), accounts)
        req_dict = accounts.get("acc_2", {})
        xt_trader = req_dict.get("xt_trader")
        acc = req_dict.get("account")
        is_test = req_dict.get("is_test", "0")
        self.is_test = is_test
        # acc_name = req_dict.get("acc_name")
        ## ������Ч�ٻس�
        if len(self.pools_list) == 0:
            return json.dumps({"warn":[{"mark":"pools_list is empty."}]})

        ## ����ʱ��
        cur_time = datetime.datetime.now().time()
        gy_time = datetime.datetime.strptime("22:01", "%H:%M").time()
        jj_start_time = datetime.datetime.strptime("09:15", "%H:%M").time()
        jj_end_time = datetime.datetime.strptime("09:19", "%H:%M").time()
        start_time = datetime.datetime.strptime("09:15", "%H:%M").time()
        open_time = datetime.datetime.strptime("09:30", "%H:%M").time()
        mid_start_time = datetime.datetime.strptime("11:30", "%H:%M").time()
        mid_end_time = datetime.datetime.strptime("13:01", "%H:%M").time()
        end_time = datetime.datetime.strptime("14:55", "%H:%M").time()
        is_trade_time = start_time <= cur_time <= mid_start_time or mid_end_time <= cur_time <= end_time
        is_jj_time = jj_start_time <= cur_time <= jj_end_time
        is_open_time = cur_time >= open_time
        ## ���ڽ���ʱ�䲻����
        if not is_trade_time and self.is_test == "0":
            return json.dumps({"warn":[{"mark":"is_not_trade_time."}]})

        # ��ͬ�����ɽ���������Ϊ����
        eff_stock_list = list()
        limit_1_index = 0
        limit_2_index = 0
        for code in self.pools_list:
            content = self.stock_dict.get(code, {})
            # print('===========', code, content)
            stock_code = content.get("code", "")
            if len(stock_code) == 0:
                continue
            limit_up_days = content.get("continuous_up_limit_days", 0)
            if limit_up_days == 0:
                continue
            if limit_up_days == 1 and limit_1_index < 2:
                limit_1_index += 1
                eff_stock_list.append(stock_code)

            elif limit_up_days == 2 and limit_2_index < 2:
                limit_2_index += 1
                eff_stock_list.append(stock_code)

        ## ������Ч�Ľ����
        if len(eff_stock_list) == 0:
           return json.dumps({"msg": [{"mark": "eff_stock_list is empty."}]})
        print(f"[DEBUG]eff_stock_size={len(eff_stock_list)};pools_list={eff_stock_list}")

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

            code = has_stock_code.split('.')[0]
            industry = self.stock_dict.get(code, {}).get("industry", "")
            if industry in self.industry_num_map:
                self.industry_num_map[industry] += 1
            else:
                self.industry_num_map[industry] = 1
        subscribe_whole_list = list(set(has_stock_list + eff_stock_list))
        print(f"[DEBUG]subscribe_whole_list={subscribe_whole_list}")

        # ע��ȫ�ƻص�����
        # ������һ���յ��б����洢���ؽ��
        final_ret_list = []
        loop = asyncio.get_event_loop()
        # ע��ȫ�ƻص�����
        def callback(data):
            ret = self.price_update_callback(data, xt_trader, acc, eff_stock_list, is_jj_time, is_open_time)
            if ret is not None:
                final_ret_list.extend(ret)

        xtdata.subscribe_whole_quote(subscribe_whole_list, callback=callback)

        # ����������xtdata.run()�������ں�̨�߳�������
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, xtdata.run)













        # # ע��ȫ�ƻص�����
        # # ������һ���յ��б����洢���ؽ��
        # final_ret_list = []
        #
        # # ע��ȫ�ƻص�����
        # def callback(data):
        #     ret = self.price_update_callback(data, xt_trader, acc, eff_stock_list)
        #     if ret is not None:
        #         final_ret_list.extend(ret)
        #
        # xtdata.subscribe_whole_quote(subscribe_whole_list, callback=callback)
        # xtdata.run()
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

def is_price_declining(prices, window=3):
    """������window��ʱ����Ƿ�����µ�"""
    if len(prices) < window:
        return False
    for i in range(1, window):
        if prices[-i] >= prices[-i-1]:
            return False
    return True

def get_yesterday_date():
    """ ��ȡ��Aǰһ�������յ����� """
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    # print(today)
    previous_trade_date = a_trade_calendar.get_pre_trade_date(today).replace('-', '')
    # print(previous_trade_date)
    #return (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y%m%d')
    return previous_trade_date

if __name__ == "__main__":
    a = Dragon_V5(config="../conf/v1.ini")
