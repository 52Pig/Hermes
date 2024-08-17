#coding=utf8


import  random
from xtquant.xttrader import XtQuantTrader
#from xtquant_240613.xttrader import XtQuantTrader
path = 'D:/tool/gjqmt_client/userdata_mini/'
session_id = int(random.randint(100000, 999999))
xt_trader = XtQuantTrader(path, session_id)

xt_trader.start()
connect_result = xt_trader.connect()
print(connect_result)


from xtquant.xttype import StockAccount
acc = StockAccount('8886086606')
subsribe_result = xt_trader.subscribe(acc)
print(subsribe_result)

#下单
from xtquant import xtconstant
stock_code = '600138.SH'
order_id = xt_trader.order_stock(acc, stock_code, xtconstant.STOCK_BUY, 100, xtconstant.FIX_PRICE, 9.7)
print(order_id)

#撤单
xt_trader.cancel_order_stock(acc, order_id)