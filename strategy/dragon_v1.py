#coding=gbk

import time
import datetime
import a_trade_calendar
from xtquant import xtdata
from base_strategy import BaseStrategy

'''
策略逻辑

条件：
1，前一个交易日涨停的沪深
2，排除ST内容
3，排除单价在3元以下
4，开盘价在x%以上内容


买入：
1，每分钟股价在2%以上，买入概率随有效分钟数增加0.01
2，若有大单买入，而增加

出逃：
1，一天最多允许买入一支，行情结束再买下一支？
'''


class Dragon_V1(BaseStrategy):
    def __init__(self, config):
        pass





    def do(self, accounts):
        print()




def get_index_stocks(index_code):
    """获取沪深300指数的成分股列表"""
    return xtdata.get_stock_list_in_sector(index_code)


def get_yesterday_date():
    """ 获取大A前一个交易日的日期 """
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    # print(today)
    previous_trade_date = a_trade_calendar.get_pre_trade_date(today).replace('-', '')
    # print(previous_trade_date)
    #return (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y%m%d')
    return previous_trade_date

def get_today_date():
    """ 定义获取今日日期的函数"""
    return datetime.datetime.now().strftime('%Y%m%d')

def get_ztgp_stocks(index_stocks, yesterday):
    """定义获取昨日涨停股票的函数
     存储涨停股票的列表
    """
    ztgp_stocks = []
    # 获取沪深300指数的成分股昨日的行情数据
    for stock in index_stocks:
        stock_code = ''
        if stock.endswith(".SH") or stock.endswith(".SZ"):
            stock_code = stock
        elif stock.isdigit():
            stock_code = stock + '.SH'
        else:
            continue
            #stock_code = stock + ".SH" if stock.isdigit() else stock + ".SZ"

        ## 排除ST的股票,创业板，
        instrument_detail = xtdata.get_instrument_detail(stock_code)
        #if '300615' in stock_code:
        #print('[DEBUG]instrument_detail=',instrument_detail)
        stock_name = ''
        if instrument_detail is not None:
            stock_name = instrument_detail.get("InstrumentName", "")
            if "ST" in stock_name:
                #print("[DEBUG]filter_ST=", stock_code, stock_name)
                continue
        #if 'GEM' in instrument_detail.get('Market', '').upper() or '创业板' in instrument_detail.get('Market', ''):
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

        # 检查是否存在涨停的情况
        if data and data[stock_code].size != 0:
            close_price = data[stock_code].iloc[0]['close']
            open_price = data[stock_code].iloc[0]['open']
            if stock_code == '600843.SH':
                print(stock_code, stock_name, open_price, close_price)
            # 排除昨收价在3.0元以下
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
    """# 定义获取股票当前价格的函数
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
    """获取股票开盘价的函数"""
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
    """"# 定义检查并执行卖出操作的函数
    """
    current_price = xtdata.get_market_data(
        stock_code,
        field_list=['price'],
        period='d',
        start_time=get_current_time(),
        count=1
    ).iloc[0]['price']

    if current_price is not None and current_price < opening_price:
        print(f"{get_current_time()} - 股票代码: {stock_code}, 当前价格: {current_price}, 开盘价: {opening_price}, 执行卖出操作。")
        # 执行卖出操作（这里只是打印信息，实际中应调用API执行卖出）
        # xtdata.sell_stock(stock_code, amount)


if __name__ == "__main__":
    # 获取沪深300指数代码
    index_code = '沪深A股'
    # xtdata._download_history_data(index_code, period='1d', start_time='20240801', end_time='20240811')
    # xtdata.download_sector_data()
    # xtdata.get_stock_list_in_sector(index_code)

    # 获取昨日日期
    # yesterday = get_yesterday_date()
    yesterday = '20240927'
    # 获取当日日期
    # trade_date = get_today_date()
    trade_date = '20240930'
    print('[DEBUG]yesterday=', yesterday)
    print('[DEBUG]trade_date=', trade_date)
    # 获取沪深300指数的成分股列表
    index_stocks = get_index_stocks(index_code)
    print("[DEBUG]hs=", index_stocks)
    # 下载每支股票数据
    xtdata.download_history_data2(stock_list=index_stocks, period='1d', start_time=yesterday, end_time=trade_date)

    # 获取并打印昨日涨停股票
    ztgp_stocks = get_ztgp_stocks(index_stocks, yesterday)
    # print('[DEBUG]pools=', ztgp_stocks)
    # 写出日志
    fw_file = open('logs/pools_' + trade_date + ".txt", 'w')
    for stock in ztgp_stocks:
        row_line = f"股票代码: {stock[0]}, 昨日收盘价: {stock[1]}, 昨日开盘价：{stock[2]}, 名称: {stock[3]}, 昨日收益:{stock[4]}"
        print(row_line)
        fw_file.write(row_line+'\n')
    # 每分钟检查一次符合条件的股票价格

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
                print(f"{get_current_time()} - 股票代码: {stock_code}, 当前价格: {current_price}, 昨收价格：${last_close_price}")
                # 卖出
                if (current_price - last_close_price) / last_close_price < -0.027:
                    print("sell out")
                ## 开盘价在2%以下则忽略此股
                if (openning_price - last_close_price) / last_close_price < 0.031:
                    print("sell out")
                # 买入
                # 1，每分钟股价在2 % 以上，买入概率随有效分钟数增加0.01
                if (current_price - last_close_price) / last_close_price > 0.021:
                    if stock_times_dict.get(stock_code, -1) == -1:
                        stock_times_dict[stock_code] = 0.01
                    else:
                        stock_times_dict[stock_code] += 0.01
                #
                # 2，(当前分钟 - 前一分钟价格) / 前一分钟价格 > 0.022,则买入
                #

                # 3，若有大单买入，而增加买入权重
                # if (current_price - last_close_price) / last_close_price > 0.02:


        time.sleep(60)