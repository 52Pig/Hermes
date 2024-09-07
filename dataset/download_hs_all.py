#coding=utf8

'''
下载沪深最近6个月行情数据

output schema:
stock_code, tm, pprice, high, low, close, volume, amount
'''

import sys
from xtquant import xtdata
# 获取沪深300指数代码
index_code = '沪深A股'
# xtdata._download_history_data(index_code, period='1d', start_time='20240801', end_time='20240811')
# xtdata.download_sector_data()
# xtdata.get_stock_list_in_sector(index_code)

# 获取昨日日期
# yesterday = get_yesterday_date()
# yesterday = '20240827'
yesterday = '20240501'
# 获取当日日期
# trade_date = get_today_date()
trade_date = '20240906'
print('[DEBUG]yesterday=', yesterday)
print('[DEBUG]trade_date=', trade_date)
# 获取沪深的成分股列表
index_stocks = xtdata.get_stock_list_in_sector(index_code)
print("[DEBUG]hs=", len(index_stocks), index_stocks)

# 下载每支股票数据
xtdata.download_history_data2(stock_list=index_stocks, period='1m', start_time=yesterday, end_time=trade_date)



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
fw_file = open('./hs_all_data.txt', 'w')
for stock in index_stocks:
    # print(xtdata.get_field_list())

    data = xtdata.get_market_data_ex(
        stock_list=[stock],
        field_list=['open', 'high', 'low', 'close', 'volume', 'amount', 'lastPrice', 'pre_close', 'change',
              'pct_chg', 'turnover', 'mkt_cap', 'pe', 'pb', 'ps', 'pcf_nf', 'pcf', 'ps_tf', 'dv_ratio', 'eps',
              'roe', 'roa', 'current_ratio', 'quick_ratio', 'cash_ratio'],
        period='1m',
        start_time=yesterday,
        end_time=trade_date,
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
            row_line = '\t'.join((stock_code, tm, str(pprice), str(high), str(low), str(close), str(volume), str(amount)))
            fw_file.write(row_line+"\n")
    print("[DEBUG]download finish!")
    # break