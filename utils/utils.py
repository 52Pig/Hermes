
import numpy as np
from datetime import datetime
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

def get_latest_price(stock_code):
    """# 定义获取股票当前价格的函数
    """
    data = xtdata.get_market_data_ex(
        stock_list=[stock_code],
        field_list=['time', 'close'],
        period='1m',
        count=1
    )
    if data is None:
        return None
    # print(data)
    df = data[stock_code]
    # 获取最大时间戳的行
    tdata = df.loc[df['time'].idxmax()]
    # print(tdata)
    if 'close' in tdata:
        return tdata['close']
    else:
        return None


