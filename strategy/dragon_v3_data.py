#coding=gbk
import sys
sys.path.append('../')
import json
import random
import datetime
import configparser
import a_trade_calendar
from utils import utils

from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount
from xtquant import xtdata
from xtquant import xtconstant
'''
离线每天筛选出股票池，用于线上做加载
'''

class Dragon_V3_Data():
    def __init__(self):
        # config = configparser.ConfigParser()
        # config_path = 'conf/config.ini'
        # config.read(config_path)
        # mini_path = config.get("mini_path")
        # acc_name = config.get("acc_name")
        # session_id = int(random.randint(100000, 999999))
        # xt_trader = XtQuantTrader(mini_path, session_id)
        # xt_trader.start()
        # print('acc_name', acc_name)
        # connect_result = xt_trader.connect()
        # acc = StockAccount(acc_name)
        # subscribe_res = xt_trader.subscribe(acc)
        # self.xt_trader = xt_trader
        # self.acc = acc
        # print(f'[DEBUG] Account initialized, connect_status={connect_result}, subscribe_status={subscribe_res}')
        print("INIT DONE!")

    def do(self, out_file):
        print("[DEBUG]do dragon_v3_data ", utils.get_current_time())
        target_code = '沪深A股'
        # xt_trader = self.xt_trader
        # acc = self.acc
        # 查询沪深所有股票
        index_stocks = xtdata.get_stock_list_in_sector(target_code)
        ## 筛选出有效召回池
        recall_stock = list()
        for stock_code in index_stocks:
            if not stock_code.endswith(".SH") and not stock_code.endswith(".SZ"):
                continue
            ## 排除ST的股票
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
            # 排除创业板
            if stock_code.startswith("3"):
                # print("[DEBUG]filter_chuangyeban=", stock_code, stock_name)
                continue
            ## 是否停牌
            ins_status = instrument_detail.get("InstrumentStatus",0)
            if ins_status >= 1:
                print("[DEBUG]=instrumentStatus", ins_status, stock_code, stock_name)
                continue
            recall_stock.append((stock_code, stock_name))
        #print(len(recall_stock), recall_stock)

        ## 股价价格筛选
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

        # 计算最近N个交易日连板的天数
        N = 10
        last_n = utils.get_past_trade_date(N)
        #print(f'[DEBUG]last_n={last_n} ;eff_stock_list={len(eff_stock_list)};eff_stock_list={eff_stock_list}')
        pre_ztgp_stocks = get_ztgp_days(eff_stock_list, last_n)
        print(f'[DEBUG]pre_ztgp_stocks={pre_ztgp_stocks}')
        sorted_stocks = filter_and_sort_stocks(pre_ztgp_stocks)
        print(f"[DEBUG]sorted_stocks={sorted_stocks}")
        if len(sorted_stocks) == 0:
            return json.dumps({"msg":[{"mark":"sorted_stocks is empty."}]})
        fw_file = open(out_file, 'w')
        for stock_code, limit_up_days, yesterday_volume in sorted_stocks:
            row_line = '\t'.join((stock_code, str(limit_up_days), str(yesterday_volume)))
            fw_file.write(row_line + '\n')


def is_highest_bid(self, stock_code):
    """检查该股票的封单量是否是相同连板数中最高"""
    # 获取该股票的连板数
    limit_up_days = self.calculate_limit_up_days(stock_code)

    # 获取相同连板数的所有股票及其封单量
    same_limit_up_stocks = self.get_same_limit_up_stocks(limit_up_days)

    if not same_limit_up_stocks:
        return False  # 没有找到相同连板数的股票

    # 获取该股票的封单量
    current_bid_volume = self.get_current_bid_volume(stock_code)

    # 检查封单量是否为最高
    is_highest = all(current_bid_volume >= volume for _, volume in same_limit_up_stocks)
    return is_highest


def get_yesterday_date():
    """ 获取大A前一个交易日的日期 """
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    # print(today)
    previous_trade_date = a_trade_calendar.get_pre_trade_date(today).replace('-', '')
    # print(previous_trade_date)
    #return (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y%m%d')
    return previous_trade_date


def filter_and_sort_stocks(ztgp_stocks):
    """排除2天以下和5天以上涨停的股票，并按当日成交量排序"""
    filtered_stocks = []

    for stock_code, limit_up_days in ztgp_stocks:
        if 2 <= limit_up_days <= 5:
            # 获取昨日的成交量数据
            volume_data = xtdata.get_market_data_ex(
                stock_list=[stock_code],
                field_list=['time', 'volume'],
                period='1d',
                start_time='20240601'  # 根据需要调整时间范围
            )

            # 处理返回的数据
            if stock_code not in volume_data or len(volume_data[stock_code]) < 2:
                continue  # 数据不足，跳过

            stock_volume_data = volume_data[stock_code]
            yesterday_volume = stock_volume_data['volume'].iloc[-1]  # 昨日成交量
            filtered_stocks.append((stock_code, limit_up_days, yesterday_volume))

    # 按照昨日成交量从大到小排序
    sorted_stocks = sorted(filtered_stocks, key=lambda x: int(x[2]), reverse=True)
    return sorted_stocks

def get_ztgp_days(index_stocks, last_n):
    """获取昨日涨停股票及其涨停天数
    Args:
        index_stocks: 股票代码列表
        start_date: 起始计算日期，格式为'YYYYMMDD'
    Returns:
        涨停股票及其涨停天数的列表
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

        # 处理返回的数据
        if stock_code not in data or len(data[stock_code]) < 2:
            ztgp_stocks.append((stock_code, 0))  # 数据不足，返回0天数
            continue

        stock_data = data[stock_code]

        # 检查昨日是否涨停
        if (stock_data['close'].iloc[-1] - stock_data['close'].iloc[-2]) / stock_data['close'].iloc[-2] < 0.095:
            ztgp_stocks.append((stock_code, 0))  # 昨天没有涨停，返回0天数
            continue

        limit_up_count = 0
        for i in range(len(stock_data) - 1, 0, -1):  # 倒序检查
            if (stock_data['close'].iloc[i] - stock_data['close'].iloc[i - 1]) / stock_data['close'].iloc[
                i - 1] >= 0.095:
                limit_up_count += 1
            else:
                break  # 遇到不满足涨停条件，停止计数

        ztgp_stocks.append((stock_code, limit_up_count))

    return ztgp_stocks

if __name__ == "__main__":
    # 获取当前日期时间的字符串，格式为 "年-月-日 时:分:秒"
    formatted_datetime = datetime.datetime.now().strftime("%Y%m%d")
    out_file = '../logs/dragon_v3_data.' + formatted_datetime
    print("输入文件：", out_file)
    obj = Dragon_V3_Data()
    obj.do(out_file)