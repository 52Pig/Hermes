#coding=gbk

import time
import datetime
import a_trade_calendar
from xtquant import xtdata
from base_strategy import BaseStrategy

'''
�����߼�

������
1��ǰһ����������ͣ�Ļ���
2���ų�ST����
3���ų�������3Ԫ����
4�����̼���x%��������


���룺
1��ÿ���ӹɼ���2%���ϣ������������Ч����������0.01
2�����д����룬������

���ӣ�
1��һ�������������һ֧���������������һ֧��
'''


class Dragon_V1(BaseStrategy):
    def __init__(self, config):
        pass





    def do(self, accounts):
        print()




def get_index_stocks(index_code):
    """��ȡ����300ָ���ĳɷֹ��б�"""
    return xtdata.get_stock_list_in_sector(index_code)


def get_yesterday_date():
    """ ��ȡ��Aǰһ�������յ����� """
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    # print(today)
    previous_trade_date = a_trade_calendar.get_pre_trade_date(today).replace('-', '')
    # print(previous_trade_date)
    #return (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y%m%d')
    return previous_trade_date

def get_today_date():
    """ �����ȡ�������ڵĺ���"""
    return datetime.datetime.now().strftime('%Y%m%d')

def get_ztgp_stocks(index_stocks, yesterday):
    """�����ȡ������ͣ��Ʊ�ĺ���
     �洢��ͣ��Ʊ���б�
    """
    ztgp_stocks = []
    # ��ȡ����300ָ���ĳɷֹ����յ���������
    for stock in index_stocks:
        stock_code = ''
        if stock.endswith(".SH") or stock.endswith(".SZ"):
            stock_code = stock
        elif stock.isdigit():
            stock_code = stock + '.SH'
        else:
            continue
            #stock_code = stock + ".SH" if stock.isdigit() else stock + ".SZ"

        ## �ų�ST�Ĺ�Ʊ,��ҵ�壬
        instrument_detail = xtdata.get_instrument_detail(stock_code)
        #if '300615' in stock_code:
        #print('[DEBUG]instrument_detail=',instrument_detail)
        stock_name = ''
        if instrument_detail is not None:
            stock_name = instrument_detail.get("InstrumentName", "")
            if "ST" in stock_name:
                #print("[DEBUG]filter_ST=", stock_code, stock_name)
                continue
        #if 'GEM' in instrument_detail.get('Market', '').upper() or '��ҵ��' in instrument_detail.get('Market', ''):
        #    print("[DEBUG]filter_chuangyeban=", stock_code, stock_name)
        #    continue
        if stock_code.startswith("3"):
            #    print("[DEBUG]filter_chuangyeban=", stock_code, stock_name)
            continue
        data = xtdata.get_market_data_ex(
            stock_list=[stock_code],
            field_list=['close', 'open'],
            period='1d',
            start_time=yesterday,
            end_time=yesterday,
            count=1
        )
        #print('[DEBUG]data=', data)

        # ����Ƿ������ͣ�����
        if data and data[stock_code].size != 0:
            close_price = data[stock_code].iloc[0]['close']
            open_price = data[stock_code].iloc[0]['open']
            if stock_code == '600843.SH':
                print(stock_code, stock_name, open_price, close_price)
            # �ų����ռ���3.0Ԫ����
            if close_price < 2.8 or close_price > 50.0:
                # print('[DEBUG]filter_close_price<3=', stock_code, close_price, open_price)
                continue
            last_revenue_rate = (close_price - open_price) / open_price
            if last_revenue_rate >= 0.095:
                ztgp_stocks.append((stock_code, close_price, open_price, stock_name, last_revenue_rate))

    return ztgp_stocks


def get_current_time():
    return datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')


def get_current_price(stock_code):
    """# �����ȡ��Ʊ��ǰ�۸�ĺ���
    """
    data = xtdata.get_market_data(
        stock_code=stock_code,
        field_list=['price'],
        period='1d',
        start_time=get_current_time(),
        count=1
    )
    return data.iloc[0]['price'] if data is not None and data.empty == False else None


def get_opening_price(stock_code, trade_date):
    """��ȡ��Ʊ���̼۵ĺ���"""
    data = xtdata.get_market_data_ex(
        field_list=['open'],
        stock_list=[stock_code],
        period='1d',
        start_time=trade_date,
        count=1
    )

    print(data)
    return data.iloc[0]['open'] if not data.empty else None


def check_and_sell(stock_code, opening_price):
    """"# �����鲢ִ�����������ĺ���
    """
    current_price = xtdata.get_market_data(
        stock_code,
        field_list=['price'],
        period='d',
        start_time=get_current_time(),
        count=1
    ).iloc[0]['price']

    if current_price is not None and current_price < opening_price:
        print(f"{get_current_time()} - ��Ʊ����: {stock_code}, ��ǰ�۸�: {current_price}, ���̼�: {opening_price}, ִ������������")
        # ִ����������������ֻ�Ǵ�ӡ��Ϣ��ʵ����Ӧ����APIִ��������
        # xtdata.sell_stock(stock_code, amount)


if __name__ == "__main__":
    # ��ȡ����300ָ������
    index_code = '����A��'
    # xtdata._download_history_data(index_code, period='1d', start_time='20240801', end_time='20240811')
    # xtdata.download_sector_data()
    # xtdata.get_stock_list_in_sector(index_code)

    # ��ȡ��������
    # yesterday = get_yesterday_date()
    yesterday = '20240927'
    # ��ȡ��������
    # trade_date = get_today_date()
    trade_date = '20240930'
    print('[DEBUG]yesterday=', yesterday)
    print('[DEBUG]trade_date=', trade_date)
    # ��ȡ����300ָ���ĳɷֹ��б�
    index_stocks = get_index_stocks(index_code)
    print("[DEBUG]hs=", index_stocks)
    # ����ÿ֧��Ʊ����
    xtdata.download_history_data2(stock_list=index_stocks, period='1d', start_time=yesterday, end_time=trade_date)

    # ��ȡ����ӡ������ͣ��Ʊ
    ztgp_stocks = get_ztgp_stocks(index_stocks, yesterday)
    # print('[DEBUG]pools=', ztgp_stocks)
    # д����־
    fw_file = open('logs/pools_' + trade_date + ".txt", 'w')
    for stock in ztgp_stocks:
        row_line = f"��Ʊ����: {stock[0]}, �������̼�: {stock[1]}, ���տ��̼ۣ�{stock[2]}, ����: {stock[3]}, ��������:{stock[4]}"
        print(row_line)
        fw_file.write(row_line+'\n')
    # ÿ���Ӽ��һ�η��������Ĺ�Ʊ�۸�

    while True:
        stock_times_dict = dict()
        for pools in ztgp_stocks:
            stock_code = pools[0]
            last_close_price = pools[1]
            last_open_price = pools[2]
            stock_name = pools[3]
            openning_price = get_opening_price(stock_code, trade_date)
            current_price = get_current_price(stock_code)

            if current_price is not None and last_close_price is not None:
                print(f"{get_current_time()} - ��Ʊ����: {stock_code}, ��ǰ�۸�: {current_price}, ���ռ۸�${last_close_price}")
                # ����
                if (current_price - last_close_price) / last_close_price < -0.027:
                    print("sell out")
                ## ���̼���2%��������Դ˹�
                if (openning_price - last_close_price) / last_close_price < 0.031:
                    print("sell out")
                # ����
                # 1��ÿ���ӹɼ���2 % ���ϣ������������Ч����������0.01
                if (current_price - last_close_price) / last_close_price > 0.021:
                    if stock_times_dict.get(stock_code, -1) == -1:
                        stock_times_dict[stock_code] = 0.01
                    else:
                        stock_times_dict[stock_code] += 0.01
                #
                # 2��(��ǰ���� - ǰһ���Ӽ۸�) / ǰһ���Ӽ۸� > 0.022,������
                #

                # 3�����д����룬����������Ȩ��
                # if (current_price - last_close_price) / last_close_price > 0.02:


        time.sleep(60)