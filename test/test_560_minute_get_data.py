# stock_code = '603883.SH'
# stock_code = '000948.SZ'
# xtdata.subscribe_quote(stock_code, period=period, count=-1)  # 设置count = -1来取到当天所有实时行情
# 下载每支股票数据




from xtquant import xtdata

def get_xtdata_1():
    stock_code = '000560.SZ'
    #field_list = ['time', 'open', 'high', 'low', 'close', 'volume', 'lastPrice']
    field_list = ['time', 'open', 'high', 'close', 'lastPrice']
    xtdata.download_history_data(stock_code, '1m', '20240101')
    #kline_data = xtdata.get_market_data_ex(field_list=[], stock_list=[stock_code], period='1m',start_time='20240101')
    #kline_data = xtdata.get_market_data_ex(field_list=[], stock_list=[stock_code], period='1m',start_time='20240923093000')
    kline_data = xtdata.get_market_data_ex(field_list, stock_list=[stock_code], period='1m',start_time='20240923093000')
    print(kline_data)
    # print(xtdata.data_dir)

def get_xtdata_2():
    stock_code = '000560.SZ'
    xtdata.download_history_data(stock_code, '1m', '20240901')
    data = xtdata.get_market_data_ex(
        ['time', 'close'],
        stock_list=[stock_code],
        period='1m',
        start_time='20240926093000'
    )
    # kline_data = xtdata.get_market_data_ex(field_list, stock_list=[stock_code], period='1m',
    #                                        start_time='20240923093000')

    print('[utils]get_latest_price, data=', data)

if __name__ == "__main__":
    get_xtdata_2()
    # get_xtdata_1()