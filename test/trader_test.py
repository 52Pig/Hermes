#coding=utf8


import  random
from xtquant.xttrader import XtQuantTrader
#from xtquant_240613.xttrader import XtQuantTrader
path = 'D:/tool/gj_client/userdata_mini/'
session_id = int(random.randint(100000, 999999))
xt_trader = XtQuantTrader(path, session_id)

xt_trader.start()
connect_result = xt_trader.connect()
print(connect_result)


from xtquant.xttype import StockAccount
acc = StockAccount('8886086606')
subsribe_result = xt_trader.subscribe(acc)
print(subsribe_result)

# #下单
# from xtquant import xtconstant
# stock_code = '600138.SH'
# order_id = xt_trader.order_stock(acc, stock_code, xtconstant.STOCK_BUY, 100, xtconstant.FIX_PRICE, 9.7)
# print(order_id)
#
# #撤单
# xt_trader.cancel_order_stock(acc, order_id)


##
import time
acc_info = xt_trader.query_stock_asset(acc)
print(dir(acc_info))
print('[DEBUG]acc_id=', acc_info.account_id)
print('[DEBUG]acc_type=', acc_info.account_type)
print('[DEBUG]可用金额=cash=', acc_info.cash)
print('[DEBUG]冻结金额=frozen_cash=', acc_info.frozen_cash)
print('[DEBUG]m_dCash=', acc_info.m_dCash)
print('[DEBUG]m_dFrozenCash=', acc_info.m_dFrozenCash)
print('[DEBUG]m持仓市值=m_dMarketValue=', acc_info.m_dMarketValue)
print('[DEBUG]m总资产=m_dTotalAsset=', acc_info.m_dTotalAsset)
print('[DEBUG]m_nAccountType=', acc_info.m_nAccountType)
print('[DEBUG]m_strAccountID=', acc_info.m_strAccountID)
print('[DEBUG]持仓市值=market_value=', acc_info.market_value)
print('[DEBUG]总资产=total_asset=', acc_info.total_asset)

has_stock_list = xt_trader.query_stock_positions(acc)
for stock in has_stock_list:
    print(dir(stock))
    print('m_strAccountID=', stock.m_strAccountID)
    print('m_strStockCode=', stock.m_strStockCode)
    print('m_strStockCode1=', stock.m_strStockCode1)
    print('持仓总市值=market_value=', stock.market_value)
    print('持仓数量=on_road_volume=', stock.on_road_volume)
    print('开仓价=open_price=', stock.open_price)
    print('持仓股票=stock_code=', stock.stock_code)
    print('stock_code1=', stock.stock_code1)
    print('持仓数量=volume=', stock.volume)
    print('昨日持仓数量=yesterday_volume=', stock.yesterday_volume)

#['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'account_id', 'account_type', 'avg_price', 'can_use_volume', 'direction', 'frozen_volume', 'm_dAvgPrice', 'm_dMarketValue', 'm_dOpenPrice', 'm_nAccountType', 'm_nCanUseVolume', 'm_nDirection', 'm_nFrozenVolume', 'm_nOnRoadVolume', 'm_nVolume', 'm_nYesterdayVolume', 'm_strAccountID', 'm_strStockCode', 'm_strStockCode1', 'market_value', 'on_road_volume', 'open_price', 'stock_code', 'stock_code1', 'volume', 'yesterday_volume']
# m_strAccountID= 8886086606
# m_strStockCode= 000858.SZ
# m_strStockCode1=
# market_value= 12111.0
# on_road_volume= 0
# open_price= 131.05
# stock_code= 000858.SZ
# stock_code1=
# volume= 100
# yesterday_volume= 100


import backtrader


from xtquant import xtdata
from datetime import datetime
import time
import pandas as pd

code = '000560.SZ'
# 初始化一个空的 DataFrame
df_columns = ['code', 'time', 'open', 'close', 'high', 'low', 'volume']
df = pd.DataFrame(columns=df_columns)

def on_data (datas):
    global df  # 使用全局变量以便更新 DataFrame
    tick_time = datas[code]['time']
    timestamp_seconds = tick_time / 1000
    readable_time = datetime.fromtimestamp(timestamp_seconds).strftime('%Y-%m-%d %H:%M:%S.%f')
    # 获取当前时间戳（秒级）
    current_timestamp_seconds = time.time()
    current_readable_time = datetime.fromtimestamp(current_timestamp_seconds).strftime('%Y-%m-%d %H:%M:%S.%f')
    print(readable_time)
    print(current_readable_time)
    print(datas)
    if code in datas:
        tm = datas[code]['time']
        open = datas[code]['open']
        close = datas[code]['lastPrice']
        high = datas[code]['high']
        low = datas[code]['low']
        volume = datas[code]['volume']
        print(tm, open, high, low, close, volume)

        # 将数据添加到 DataFrame
        new_row = pd.DataFrame([{
            'code': code,
            'time': readable_time,
            'open': open,
            'close': close,
            'high': high,
            'low': low,
            'volume': volume
        }])

        df = pd.concat([df, new_row], ignore_index=True)
        print(df)


xtdata.subscribe_whole_quote(code_list=[code], callback=on_data)
#xtdata.subscribe_whole_quote(code_list=[code])
data = xtdata.get_market_data(['time', 'open', 'high', 'low', 'close', 'volume'], [code], period='1m', start_time='20240918')
# data = xtdata.get_market_data(['time', 'open', 'high', 'low', 'close', 'volume'], [code], period='1m')
print('result=====',data)
# {'000560.SZ': {'time': 1726210800000, 'lastPrice': 2.45, 'open': 2.4, 'high': 2.5100000000000002, 'low': 2.38, 'lastClose': 2.38, 'amount': 515111100.0, 'volume': 2100053, 'pvolume': 210005323, 'stockStatus': 0, 'openInt': 15, 'transactionNum': 0, 'lastSettlementPrice': 0.0, 'settlementPrice': 0.0, 'pe': 0.0, 'askPrice': [2.45, 2.46, 2.47, 0.0, 0.0], 'bidPrice': [2.44, 2.43, 2.42, 0.0, 0.0], 'askVol': [8792, 43077, 22965, 0, 0], 'bidVol': [21723, 20008, 11821, 0, 0], 'volRatio': 0.0, 'speed1Min': 0.0, 'speed5Min': 0.0}}
xtdata.run()