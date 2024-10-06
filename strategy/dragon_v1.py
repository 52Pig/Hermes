#coding=gbk

import time
import datetime
import a_trade_calendar
from xtquant import xtdata
from xtquant import xtconstant
from base_strategy import BaseStrategy
from utils import utils
'''
策略逻辑

条件：
1，前一个交易日涨停的沪深
2，排除ST内容
3，排除单价在2元以下,40元以上
4，开盘价在x%以上内容


买入：
1，每分钟股价在2%以上，买入概率随有效分钟数增加0.01
2，若有大单买入，而增加

出逃：
1，一天最多允许买入一支，行情结束再买下一支？
'''


class Dragon_V1(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)

    def do(self, accounts):
        print("[DEBUG]do dragon_v1 ", utils.get_current_time(), accounts)
        target_code = '沪深A股'
        req_dict = accounts.get("acc_1", {})
        xt_trader = req_dict.get("xt_trader")
        acc = req_dict.get("account")
        # acc_name = req_dict.get("acc_name")

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
            if instrument_detail is not None:
                stock_name = instrument_detail.get("InstrumentName", "")
                if "ST" in stock_name:
                    # print("[DEBUG]filter_ST=", stock_code, stock_name)
                    continue
            # 排除创业板
            if stock_code.startswith("3"):
                # print("[DEBUG]filter_chuangyeban=", stock_code, stock_name)
                continue
            recall_stock.append((stock_code, stock_name))
        #print(len(recall_stock), recall_stock)

        ## 股价价格筛选
        # xtdata.download_history_data(stock_code, '1m', '20240601')
        xtdata.download_history_data2(recall_stock, period='1m', start_time='20240601')

        eff_stock_list = list()
        for stock_code, stock_name in recall_stock:
            latest_price = utils.get_latest_price(stock_code)
            if latest_price < 2.0 or latest_price > 40.0:
                print("[DEBUG]filter latest price", stock_code, stock_name)
                continue
            eff_stock_list.append(stock_code)

        # 计算最近N个交易日连板的天数
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

        # 挂隔夜单买入
        gy_order_id = ''
        if cur_time == gy_time:
            ## 买入委托
            action_name = "buy"
            # 查询账户余额
            acc_info = xt_trader.query_stock_asset(acc)
            cash = acc_info.cash
            # 查询当前股价
            current_price = utils.get_latest_price(stock_code)
            ## 若账户余额> 股票价格*100，则买入
            # 下单
            if current_price is not None:
                if cash >= current_price * 100:
                    order_id = xt_trader.order_stock(acc, stock_code, xtconstant.STOCK_BUY, 100,
                                                     xtconstant.FIX_PRICE, current_price)
                    gy_order_id = order_id
        elif cur_time == jj_time:
            ## 检查该股票的封单量是否相同连板数中最高
            is_highest = is_highest_bid(stock_code)
            if not is_highest:
                action_name = 'cancel'
                ## 取消买入委托的订单
                xt_trader.cancel_order_stock(acc, gy_order_id)

        elif start_time <= cur_time <= mid_start_time or mid_end_time <= cur_time <= end_time:
            # 检查当前时间是否在 9:30 到 15:00 之间
            #1，每分钟监测股价走势，若股价 < 前一日收盘价则卖出；
            #2，每分钟监测股价走势，若股价 > 前一日收盘价，计算(股价 - 前一日收盘价) / 前一日收盘价 > 5 % 则持有观察，若 < 5 % 则以当前价格卖出。
            print("action_name")
            current_price = utils.get_latest_price(stock_code)
            yesterday_close_price = utils.get_yesterday_close_price(stock_code)
            if ( current_price - yesterday_close_price ) / yesterday_close_price < 0.02:
                action_name = "sell"
                # 查询持仓市值
                acc_info = xt_trader.query_stock_asset(acc)
                marketValue = acc_info.m_dMarketValue
                # 卖出
                if current_price is not None:
                    if marketValue > 0:
                        order_id = xt_trader.order_stock(acc, stock_code, xtconstant.STOCK_SELL, 100,
                                                         xtconstant.FIX_PRICE, current_price)





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

            if len(volume_data) < 2:
                continue  # 数据不足，跳过
            yesterday_volume = volume_data['volume'][-1]  # 昨日成交量
            filtered_stocks.append((stock_code, limit_up_days, yesterday_volume))

    # 按照昨日成交量从大到小排序
    sorted_stocks = sorted(filtered_stocks, key=lambda x: x[2], reverse=True)
    return sorted_stocks


def get_ztgp_days(index_stocks, last_n):
    """定义获取昨日涨停股票的函数
     存储涨停股票的列表
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
            ztgp_stocks.append((stock_code, 0))  # 数据不足，返回0天数
            continue

        # 检查昨日是否涨停
        if (data['close'][1] - data['close'][0]) / data['close'][0] < 0.097:
            ztgp_stocks.append((stock_code, 0))  # 昨天没有涨停，返回0天数
            continue

        limit_up_count = 0
        for i in range(len(data) - 1, 0, -1):  # 倒序检查
            if (data['close'][i] - data['close'][i - 1]) / data['close'][i - 1] >= 0.097:
                limit_up_count += 1
            else:
                break  # 遇到不满足涨停条件，停止计数

        ztgp_stocks.append((stock_code, limit_up_count))

    return ztgp_stocks