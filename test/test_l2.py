
from xtquant import xtdata


stock_code = '000560.SZ'
field_list = ['time', 'price', 'volume', 'bidPrice', 'askPrice', 'bidVol', 'askVol']
xtdata.download_history_data(stock_code, '1m', '20240901')
## 逐笔成交
a = xtdata.get_l2_transaction(field_list, '000560.SZ', start_time='20240901')
print("1111=====", a)
## 逐笔快照
b = xtdata.get_l2_quote(field_list, stock_code, start_time='20240910')
print("22222=====", b)
## 逐笔委托
c = xtdata.get_l2_order(field_list, stock_code, start_time='20240910')
print("33333=====", c)