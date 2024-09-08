# coding=utf8

'''
下载沪深最近6个月行情数据

output schema:
stock_code, tm, pprice, high, low, close, volume, amount
'''

'''
    open：开盘价
    high：最高价
    low：最低价
    close：收盘价
    volume：成交量
    amount：成交金额
    pre_close：上一个交易日的收盘价
    change：价格变动
    pct_chg：涨跌幅
    turnover：换手率
    mkt_cap：市值
    流通市值：流通市值
    pe：市盈率
    pb：市净率
    ps：市销率
    pcf_nf：市现率
    pcf：价格/自由现金流
    ps_tf：价格/销售收入
    dv_ratio：股息率
    eps：每股收益
    roe：净资产收益率
    roa：总资产收益率
    current_ratio：流动比率
    quick_ratio：速动比率
    cash_ratio：现金比率
    ic_ratio：利息保障倍数
    inv_turnover_ratio：存货周转率
    ar_turnover_ratio：应收账款周转率
    fixed_assets_turnover_ratio：固定资产周转率
    asset_turnover_ratio：资产周转率
    equity_turnover_ratio：股东权益周转率
    net_profit_margin：销售净利率
    gross_profit_margin：销售毛利率
    roe：净资产收益率
    roa：总资产收益率
    '''
import os
import sys
from xtquant import xtdata

import xtquant as xq
from datetime import datetime


def is_date_in_range(date, start_date, end_date):
    """
    判断给定的日期是否在指定的日期范围内，包括起始日期和结束日期。
    参数:
    date: 待检查的日期，可以是字符串或 datetime 对象
    start_date: 起始日期，可以是字符串或 datetime 对象
    end_date: 结束日期，可以是字符串或 datetime 对象

    返回:
    如果日期在范围内，返回 True；否则返回 False
    """

    # 如果输入是字符串类型，先转换为 datetime 对象
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y%m%d')
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y%m%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y%m%d')
    # 判断日期是否在范围内
    return start_date <= date <= end_date


def get_trading_days(start_date, end_date):
    """
    获取指定日期范围内的所有A股交易日。

    参数:
    start_date (str): 起始日期，格式为 'YYYYMMDD'
    end_date (str): 结束日期，格式为 'YYYYMMDD'

    返回:
    list: 交易日列表
    """
    # 将字符串日期转换为datetime对象
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')
    # 获取交易日历
    # xtdata.get_trading_dates()
    # xtdata.get_trading_calendar()
    # xtdata.get_trading_time(stock_code)
    # print(xtdata.get_market_last_trade_date("SH"))
    # xtdata.download_holiday_data()
    # print(xtdata.get_trading_calendar("SH", start_time='20230103', end_time='20240103'))
    aa = xtdata.get_trading_dates("SH")
    datetime_objects = [datetime.fromtimestamp(ts / 1000.0) for ts in aa]
    # 格式化日期时间为 'YYYY-MM-DD HH:MM:SS' 格式
    # formatted_datetimes = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in datetime_objects]
    formatted_dates = [dt.strftime('%Y%m%d') for dt in datetime_objects]
    formatted_dates = [dt for dt in formatted_dates if is_date_in_range(dt, start_date, end_date)]
    # 打印转换后的日期时间
    return formatted_dates


def download_hs_all_data(start_date, end_date, folder):
    # 获取沪深300指数代码
    index_code = '沪深A股'
    # xtdata._download_history_data(index_code, period='1d', start_time='20240801', end_time='20240811')
    # xtdata.download_sector_data()
    # xtdata.get_stock_list_in_sector(index_code)

    # print('[DEBUG]start_date=', start_date)
    # print('[DEBUG]trade_date=', end_date)
    # 获取沪深的成分股列表
    index_stocks = xtdata.get_stock_list_in_sector(index_code)
    print("[DEBUG]hs=", len(index_stocks), index_stocks)

    # 下载每支股票数据
    xtdata.download_history_data2(stock_list=index_stocks, period='1m', start_time=start_date, end_time=end_date)

    out_folder = os.path.join(folder, start_date)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)
    out_name = os.path.join(out_folder, 'hs_all_data.txt')
    fw_file = open(out_name, 'w')
    for stock in index_stocks:
        # print(xtdata.get_field_list())
        data = xtdata.get_market_data_ex(
            stock_list=[stock],
            field_list=['open', 'high', 'low', 'close', 'volume', 'amount', 'lastPrice', 'pre_close', 'change',
                        'pct_chg', 'turnover', 'mkt_cap', 'pe', 'pb', 'ps', 'pcf_nf', 'pcf', 'ps_tf', 'dv_ratio', 'eps',
                        'roe', 'roa', 'current_ratio', 'quick_ratio', 'cash_ratio'],
            period='1m',
            start_time=start_date,
            end_time=end_date,
            count=-1
        )

        # print(data)
        # data.to_csv('stock_data.txt')
        for stock_code, d in data.items():
            for row in d.itertuples(index=True):
                # print(row)
                # print(f"code:{stock_code}, Index:{row.Index}, Open: {row.open}, High: {row.high}, Low: {row.low}, Close: {row.close}, Volume: {row.volume}, Amount: {row.amount}")
                tm = row.Index
                pprice = round(row.open, 2)
                high = round(row.high, 2)
                low = round(row.low, 2)
                close = round(row.close, 2)
                volume = round(row.volume, 2)
                amount = round(row.amount, 2)
                row_line = '\t'.join(
                    (stock_code, tm, str(pprice), str(high), str(low), str(close), str(volume), str(amount)))
                fw_file.write(row_line + "\n")
        print(f"[DEBUG]download {stock_code} finish!")
        # break


if __name__ == "__main__":
    # start_date = '20240501'
    # end_date = '20240807'
    start_date = '20240807'
    end_date = '20240908'
    folder = 'source_data'
    # start_date = sys.argv[1]
    # end_date = sys.argv[2]
    # folder = sys.argv[3]
    trading_days = get_trading_days(start_date, end_date)
    print(trading_days)
    for td in trading_days:
        download_hs_all_data(td, td, folder)
