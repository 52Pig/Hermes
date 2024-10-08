
import numpy as np
from datetime import datetime, timedelta
from xtquant import xtdata

def to_onehot(value, num_classes):
    """
    将数值转换为One-Hot编码。

    参数:
    value -- 要转换的数值
    num_classes -- One-Hot编码的维度（类别总数）

    返回:
    onehot_encoded -- One-Hot编码的数组
    """
    # 创建一个全0的数组，长度为One-Hot编码的维度
    onehot_encoded = np.zeros(num_classes)

    # 将对应数值的索引位置设为1
    if 0 <= value < num_classes:
        onehot_encoded[value] = 1
    else:
        raise ValueError(f"Value {value} is out of range for {num_classes} classes.")
    return onehot_encoded

def parse_time(time_str):
    ''' 根据时间获取处理后的时间特征
    :param time_str:
    :return:
    '''
    # 解析时间字符串
    timestamp = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S+08:00')
    dt = datetime.strftime(timestamp, '%Y-%m-%d %H:%M:%S')
    # print(dt, time_str)
    # 计算所需的时间单位信息
    year_day = timestamp.timetuple().tm_yday  # 一年中的第几天
    year_month = timestamp.month  # 一年中的第几月
    month_day = timestamp.day  # 一月中的第几天
    week_day = timestamp.weekday() + 1  # 一星期中的第几天（星期一为1）
    hour_of_day = timestamp.hour  # 一天中的第几小时
    minute_of_hour = timestamp.minute  # 一小时中的第几分钟

    return dt, year_day, year_month, month_day, week_day, hour_of_day, minute_of_hour

def get_latest_price(stock_code, is_download=False):
    """# 定义获取股票当前价格的函数
    """
    if is_download:
        xtdata.download_history_data(stock_code, '1m', '20240901')
    data = xtdata.get_market_data_ex(
        ['time', 'close'],
        stock_list=[stock_code],
        period='1m',
        start_time='20240808093000')
    if data is None:
        return None
    # print('[utils]get_latest_price, data=', data)
    df = data[stock_code]
    if df is None or df.empty:
        return None
    # 获取最大时间戳的行
    tdata = df.loc[df['time'].idxmax()]
    # print('[utils]get_latest_price, tdata=', tdata)
    if 'close' in tdata:
        return tdata['close']
    else:
        return None

def get_current_time():
    return datetime.now().strftime('%Y%m%d %H:%M:%S')

def get_current_date():
    return datetime.now().strftime('%Y%m%d')

def get_past_date(days):
    # 计算过去的日期
    current_date = datetime.now()
    past_date = current_date - timedelta(days=days)
    return past_date.date()

def get_past_trade_date(n):
    """ 获取过去N个交易日
    """
    import a_trade_calendar
    # 获取今天的日期
    today = a_trade_calendar.get_latest_trade_date()
    # 计算过去N个交易日的日期
    # 假设我们要查询过去5个交易日
    trade_dates = a_trade_calendar.get_pre_trade_date(today, n)
    print("[utils]get_past_trade_date:today=", today, ";trade_dates=", trade_dates)
    return trade_dates

def get_yesterday_close_price(stock_code):
    """获取股票昨日收盘价格的函数"""
    data = xtdata.get_market_data_ex(
        stock_list=[stock_code],
        field_list=['time', 'close'],
        period='1d',
        count=2  # 获取最近两天的数据
    )
    if data is None:
        return None
    df = data[stock_code]
    if df is None or df.empty:
        return None
    if len(df) < 2:
        return None  # 不足两天数据，返回None
    # 获取前一天的收盘价格
    yesterday_close = df.iloc[-2]['close']  # 倒数第二行是昨日的收盘价
    return yesterday_close


