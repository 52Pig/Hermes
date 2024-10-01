from xtquant import xtdatacenter as xtdc
from xtquant import xtdata
# 设置token
xtdc.set_token("你的token")
# 指定连接VIP行情服务器
sever_list = ['115.231.218.73:55310','115.231.218.79:55310', '218.16.123.11:55310', '218.16.123.27:55310']
xtdc.set_allow_optmize_address(sever_list)
# 开启K线全推
xtdc.set_kline_mirror_enabled(True)
# 设置start_local_service为False,使xtdc监听的端口为我们自己指定的端口
xtdc.init(start_local_service=False)
# 指定xtdc使用 58601 端口
port = 58601
try:
    xtdc.listen(port=port)
    xtdata.connect(port=port)
except Exception as e:
    if "端口" in e.args[0]:
        xtdata.reconnect(port = port)
    else:
        raise KeyboardInterrupt(f"token拉起报错，信息：{e}")

## 上面是防止部分用户不知道port被什么占用做的异常处理
# 一个用来实现主图驱动的回调函数
def handlebar(data):
    kline = xtdata.get_market_data_ex([],stock_ls[:30],period=period,count=10)
    print(kline)

stock_ls = xtdata.get_stock_list_in_sector("沪深A股") # 股票列表
period = "1m" # 数据周期
for index in range(len(stock_ls)):
    stock = stock_ls[index]
    xtdata.subscribe_quote(stock,period,count=1)
    # 这边为了演示，我只向服务器请求最近一条数据，需要当天全部的,要指定count = -1,需要当天之前的历史数据，需要进行下载

xtdata.subscribe_quote(stock,"1d",count= 1) # 要订阅多周期的话，多写一行订阅就可以了    print(f"当前订阅完成{index + 1}/{len(stock_ls)}")
xtdata.subscribe_quote("000001.SH","1d",callback=handlebar) # 定义主图，并使用handlebar实现内置中的handlebar驱动效果
xtdata.run()
