from xtquant import xtdata

def get_latest_price(stock_code):
    """# 定义获取股票当前价格的函数
    """
    data = xtdata.get_market_data_ex(
        stock_list=[stock_code],
        field_list=['time', 'lastPrice'],
        period='1m',
        count=1
    )
    if data is None:
        return None
    print(data)
    df = data[stock_code]
    # 获取最大时间戳的行
    tdata = df.loc[df['time'].idxmax()]
    print(tdata)
    if 'lastPrice' in tdata:
        return tdata['lastPrice']
    else:
        return None


stock_code = '000560.SZ'  # 这里以平安银行为例
latest_price = get_latest_price(stock_code)
if latest_price:
    print(f"股票代码：{stock_code}, 最新价格：{latest_price}")
else:
    print("获取价格失败")