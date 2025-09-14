#coding=gbk
import sys
import time

sys.path.append('../')
import json
import random
import datetime
import configparser
import a_trade_calendar
from utils import utils

from xtquant import xtdata
from xtquant import xtconstant
'''
����ÿ��ɸѡ����Ʊ�أ���������������
##schema:����,���¼�,�ǵ���,�ǵ���,�ɽ���,�ɽ���,���,���,���,��,����,����,������,��ӯ��-��̬,�о���,����ֵ,��ͨ��ֵ,����,5�����ǵ�,60���ǵ���,��������ǵ���
'''

class Dragon_V5_Data():
    def __init__(self):
        self.cur_date = '20241201'
        # print(f'[DEBUG] Account initialized, connect_status={connect_result}, subscribe_status={subscribe_res}')
        print("INIT DONE!")

    def check_zt(self, stock_code, days=20):
        stock_dict = dict()
        stock_dict[stock_code] = {}

        ## ��ѯ��Ʊ��ֵ
        market_dict = utils.get_stock_info(stock_code)
        ## ����ֵ����ֵͨ����ҵ
        total_mv = market_dict.get("total_mv", -1)
        circ_mv = market_dict.get("circ_mv", -1)
        industry = market_dict.get("industry", "")
        industry_id = market_dict.get("industry_id", "")
        pe_ttm = market_dict.get("pe_ttm", -1)
        pb = market_dict.get("pb", -1)
        # print(stock_code, total_mv, industry, industry, pe_ttm, pb)

        # ��ȡ��ʷ���ݣ���ȡ�㹻��������ȷ�����㣩
        df = utils.get_stock_hist_data_em_with_retry(
            stock_code,
            '20240901',
            # '20250417',
            '20500101',
            'D'
        )
        # 0. ����У��
        if len(df) < days:
            print(f"{stock_code} ���ݲ���,��Ҫ����{days}�����������ݣ�����ȡ��{len(df)}������������")
            return False, stock_dict
        # print(df.tail(1))
        last_days_high = df['high'].tail(days)
        latest_price = df['close'].iloc[-1]
        last_days_close = df['close'].tail(days)
        up_limit_days = 0
        continue_up_limit_days = 0
        is_up_limit_half_year = False
        half_year_days = 180
        ## 1. ���Լ�⣬����������Ƿ��й���ͣ
        for i in range(1, min(half_year_days, len(last_days_close))):
            prev_price = last_days_close.iloc[-i - 1]
            cur_price = last_days_close.iloc[-i]

            # ��ȷ��ͣ�жϣ������������룩
            # ���򣺵�ǰ���̼� >= ǰһ�����̼� * 1.099��������λС�������10%�Ƿ���
            if abs(cur_price - round(prev_price * 1.1, 2)) < 1e-6:  # ���������ȴ���
                is_up_limit_half_year = True
                break
        ## 2. ������ ������ͣ����
        # ��������ⷶΧ������Խ�磩
        max_check = min(days, len(last_days_close) - 1)  # ��Ҫ������������
        for i in range(1, max_check + 1):
            prev_price = last_days_close.iloc[-i - 1]
            cur_price = last_days_close.iloc[-i]

            # ��ȷ��ͣ�жϣ��������뵽�֣�
            expected_price = round(prev_price * 1.1, 2)
            if abs(cur_price - expected_price) < 1e-6:  # �����㾫��
                continue_up_limit_days += 1
            else:
                break  # ��������ͣ��������ֹ


        ## 3. ������ ���40������ͣ������
        # ��������ⷶΧ������Խ�磩
        max_check = min(40, len(last_days_close) - 1)  # ��Ҫ������������
        for i in range(1, max_check + 1):
            prev_price = last_days_close.iloc[-i - 1]
            cur_price = last_days_close.iloc[-i]

            # ��ȷ��ͣ�жϣ��������뵽�֣�
            expected_price = round(prev_price * 1.1, 2)
            if abs(cur_price - expected_price) < 1e-6:  # �����㾫��
                up_limit_days += 1

        # �����Ƿ���ͣ
        is_yesterday_limit_up = False
        # �����Ƿ���ͣ
        is_today_limit_up = False
        last_1d_close = last_days_close.iloc[-1]
        last_2d_close = last_days_close.iloc[-2]
        last_3d_close = last_days_close.iloc[-3]
        expected_last1d_close = round(last_2d_close * 1.1, 2)
        expected_last2d_close = round(last_3d_close * 1.1, 2)
        if abs(last_2d_close - expected_last2d_close) < 1e-6:
            is_yesterday_limit_up = True
        if abs(last_1d_close - expected_last1d_close) < 1e-6:
            is_today_limit_up = True

        # ������߼��Ƿ񴥼���ͣ��
        is_today_high_limit_up = False
        if abs(last_days_high.iloc[-1] - expected_last1d_close) < 1e-6:
            is_today_high_limit_up = True


        # print(code, up_limit_days, is_up_limit_half_year)
        ## ���days���ǵ���
        diff = (last_days_close.max() - last_days_close.min()) / last_days_close.min()
        ## 3.����ma
        ma_windows = [5, 10, 20, 30]
        for n in ma_windows:
            df[f'MA{n}'] = df['close'].rolling(window=n).mean()
            # print(df[f'MA{n}'].tail(35))
        # ȥ����ֵ
        valid_df = df.dropna(subset=['MA30']).reset_index(drop=True)

        # 4.��¼��Ϣ
        # Before accessing .iloc[-1], check if the DataFrame has data
        if not valid_df.empty:
            stock_dict[stock_code]["MA5"] = valid_df['MA5'].iloc[-1]
            stock_dict[stock_code]["MA10"] = valid_df['MA10'].iloc[-1]
            stock_dict[stock_code]["MA20"] = valid_df['MA20'].iloc[-1]
            stock_dict[stock_code]["MA30"] = valid_df['MA30'].iloc[-1]
        else:
            # Handle the case where there's no data (example: set to None)
            # stock_dict[stock_code]["MA5"] = None
            # Optionally log a warning or handle the error as needed
            print(f"Warning: No data available for {stock_code}")

        stock_dict[stock_code]["continuous_up_limit_days"] = continue_up_limit_days
        stock_dict[stock_code]["last_close"] = ','.join([str(ele) for ele in df['close'].dropna().to_list()])
        stock_dict[stock_code]["last_days_zdf"] = diff
        stock_dict[stock_code]["is_up_limit_before_half_year"] = "1" if is_up_limit_half_year == True else "0"
        stock_dict[stock_code]["is_target"] = "0"
        stock_dict[stock_code]["up_limit_days"] = up_limit_days
        stock_dict[stock_code]["total_mv"] = total_mv
        stock_dict[stock_code]["circ_mv"] = circ_mv
        stock_dict[stock_code]["industry"] = industry
        stock_dict[stock_code]["industry_id"] = industry_id
        stock_dict[stock_code]["pe_ttm"] = pe_ttm
        stock_dict[stock_code]["pb"] = pb
        stock_dict[stock_code]["is_yesterday_limit_up"] = 1 if is_yesterday_limit_up else 0
        # �����Ƿ���ͣ
        stock_dict[stock_code]["is_today_limit_up"] = 1 if is_today_limit_up else 0
        stock_dict[stock_code]["is_today_high_limit_up"] = 1 if is_today_high_limit_up else 0

        if total_mv > 260:
            ## ��ֵ̫�߲�������
            return False, stock_dict
        elif latest_price < 2.0 or latest_price > 20.0:
            ## ����̫�߲�������
            # print("[DEBUG]filter latest price", stock_code, stock_name)
            return False, stock_dict
        elif continue_up_limit_days == 2 and is_up_limit_half_year:
            stock_dict[stock_code]["is_target"] = "1"
            return True, stock_dict
        elif diff >= 0.3:
            # ����Ƿ����ܹ���
            print(f"{stock_code} ��ȥ {days}�� �ǵ��� {diff} >=0.3, min={last_days_close.min()} max={last_days_close.max()}")
            return False, stock_dict
        else:
            return False, stock_dict


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


def get_daily_last_tick_bid_ask_volume(sorted_stocks, end_date):
    # ����tick��������
    res_list = list()
    all_size = len(sorted_stocks)
    ind = 0
    for scode, limit_up_days, yesterday_volume in sorted_stocks:
        ind += 1
        is_suc = xtdata.download_history_data(scode, 'tick', start_time=end_date)
        print(f"download {ind}={scode} tick status={is_suc}, all_size={all_size}")
        data = xtdata.get_market_data_ex(
            stock_list=[scode],
            period='tick',
            start_time=end_date,
            # start_time = '20240901'
        )
        d = data[scode].tail(1).to_dict()
        bidVol = list(d['bidVol'].values())[0][0]
        askVol = list(d['askVol'].values())[0][0]
        bidPrice = list(d['bidPrice'].values())[0][0]
        askPrice = list(d['askPrice'].values())[0][0]
        # print(scode, bidVol, askVol, bidPrice, askPrice)
        res_list.append([scode, limit_up_days, yesterday_volume, bidVol, askVol, bidPrice, askPrice])
        # if ind >= 3:
        #     break
    return res_list

def filter_and_sort_stocks(ztgp_stocks):
    """�ų�1�����º�5��������ͣ�Ĺ�Ʊ���������ճɽ�������"""
    filtered_stocks = []

    for stock_code, limit_up_days in ztgp_stocks:
        if 1 <= limit_up_days:
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
            period='1d',
            start_time=start_date
        )
        # �����ص�����
        if stock_code not in data or len(data[stock_code]) < 2:
            ztgp_stocks.append((stock_code, 0))  # ���ݲ��㣬����0����
            continue

        stock_data = data[stock_code]
        # ��ȡÿ�������յ���ͣ��������
        stock_data['limit_up'] = stock_data['preClose'] * 1.1  # Ĭ����ͣ����Ϊ10%
        if stock_code.startswith('3'):  # ��ҵ���Ʊ��20% �ǵ���
            stock_data['limit_up'] = stock_data['preClose'] * 1.2
        elif stock_code.startswith('6') and 'ST' in stock_code:  # ST��Ʊ��5% �ǵ���
            stock_data['limit_up'] = stock_data['preClose'] * 1.05

        # ��������Ƿ���ͣ
        if stock_data['close'].iloc[-1] < stock_data['limit_up'].iloc[-1] - 0.01:  # ����С�����
            ztgp_stocks.append((stock_code, 0))  # ����û����ͣ������0����
            continue
        # # ��������Ƿ���ͣ
        # if (stock_data['close'].iloc[-1] - stock_data['close'].iloc[-2]) / stock_data['close'].iloc[-2] < 0.095:
        #     ztgp_stocks.append((stock_code, 0))  # ����û����ͣ������0����
        #     continue

        # ����������ͣ����
        limit_up_count = 0
        for i in range(len(stock_data) - 1, 0, -1):  # ������
            if stock_data['close'].iloc[i] >= stock_data['limit_up'].iloc[i] - 0.01:  # ������ͣ����
                limit_up_count += 1
            else:
                break  # ����δ��ͣ�����գ�ֹͣ����

        ztgp_stocks.append((stock_code, limit_up_count))
    return ztgp_stocks

if __name__ == "__main__":
    start = time.time()
    # ��ȡstock_code
    stfile = 'D:/tool/dataset/stock_pools.json'
    sc_dict = json.load(open(stfile))
    # print(sc_json.get("600259.SH"))
    # st_list = [(ele.split('.')[0], 2) for ele in sc_dict.keys()]
    # quit()

    # ��ȡ��ǰ����ʱ����ַ�������ʽΪ "��-��-�� ʱ:��:��"
    formatted_datetime = datetime.datetime.now().strftime("%Y%m%d")
    # formatted_datetime = '20250417'
    out_file = '../logs/dragon_v5_data_' + formatted_datetime
    print("����ļ���", out_file)

    # ����ļ�
    stc_json = open(out_file + '.json', 'w')
    stc_dict = dict()
    target_num = 0
    checker = Dragon_V5_Data()
    ind = 0
    for code, info_dict in sc_dict.items():
        ind += 1
        if ind % 100 == 0:
            print(f"[DEBUG]processed ind={ind}")
        ## 5��ͷ����etf
        if code.startswith("5"):
            continue
        # if code != "600622":
        #     continue
        is_cross, stock_dict = checker.check_zt(code)
        # stc_dict[code] = {'is_cross':is_cross, 'days':days, 'last_close':stock_dict.get(code)}


        stc_dict[code] = sc_dict.get(code)
        ## �Ƿ�����
        if is_cross:
            stc_dict[code]["is_target"] = '1'
            target_num += 1
            continuous_up_limit_days = stock_dict.get(code, {}).get("continuous_up_limit_days", 0)
            print(f"��Ʊ{code}״̬��{is_cross}��{continuous_up_limit_days}")
        else:
            stc_dict[code]["is_target"] = '0'
        ## �����Ƿ����ģ�����Ҫ��ʷ���ݸ���������
        stc_dict[code]["continuous_up_limit_days"] = stock_dict.get(code, {}).get("continuous_up_limit_days", "")
        stc_dict[code]["last_close"] = stock_dict.get(code, {}).get("last_close", "")
        stc_dict[code]["last_days_zdf"] = stock_dict.get(code, {}).get("last_days_zdf", 0)
        stc_dict[code]["is_up_limit_before_half_year"] = stock_dict.get(code, {}).get("is_up_limit_before_half_year", "0")
        stc_dict[code]["MA5"] = stock_dict.get(code, {}).get("MA5", 1000)
        stc_dict[code]["MA10"] = stock_dict.get(code, {}).get("MA10", 1000)
        stc_dict[code]["MA20"] = stock_dict.get(code, {}).get("MA20", 1000)
        stc_dict[code]["MA30"] = stock_dict.get(code, {}).get("MA30", 1000)

        stc_dict[code]["up_limit_days"] = stock_dict.get(code, {}).get("up_limit_days", -1)
        stc_dict[code]["total_mv"] = stock_dict.get(code, {}).get("total_mv", -1)
        stc_dict[code]["circ_mv"] = stock_dict.get(code, {}).get("circ_mv", -1)
        stc_dict[code]["industry"] = stock_dict.get(code, {}).get("industry", -1)
        stc_dict[code]["industry_id"] = stock_dict.get(code, {}).get("industry_id", -1)
        stc_dict[code]["pe_ttm"] = stock_dict.get(code, {}).get("pe_ttm", -1)
        stc_dict[code]["pb"] = stock_dict.get(code, {}).get("pb", -1)
        # �����Ƿ���ͣ
        stc_dict[code]["is_yesterday_limit_up"] = stock_dict.get(code, {}).get("is_yesterday_limit_up", -1)
        # �����Ƿ���ͣ
        stc_dict[code]["is_today_limit_up"] = stock_dict.get(code, {}).get("is_today_limit_up", -1)
        # ��������Ƿ񴥰�
        stc_dict[code]["is_today_high_limit_up"] = stock_dict.get(code, {}).get("is_today_high_limit_up", -1)

        #�н���ģ��̳���ѡ�أ�������㣺��������&&�޽���
        # �н�����ó�����������̼ۣ�����ʱ��˳������
    stc_str = json.dumps(stc_dict, ensure_ascii=False, indent=4)
    stc_json.write(stc_str + '\n')
    meta = open(out_file+'.meta', 'w')
    meta_dict = dict()
    meta_dict["pools_size"] = len(stc_dict)
    meta_dict["target_num"] = target_num
    meta_dict["spend_time"] = (time.time() - start)
    # row_line = '\t'.join(('pools_size:', str(len(stc_dict)), "; target_num:", str(target_num)))
    row_line = json.dumps(meta_dict, ensure_ascii=False, indent=4)
    # row_line = '\t'.join(('pools_size:', str(len(stc_dict)), "; target_num:", str(target_num)))
    meta.write(row_line+'\n')