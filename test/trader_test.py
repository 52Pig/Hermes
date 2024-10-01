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