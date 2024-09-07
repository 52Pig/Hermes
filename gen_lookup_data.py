#coding=utf8
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

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

def gen_lookup_data(filename, out_file):
    ''' 处理原始数据成可训练的格式
    :param filename:
    :param out_file:
    :return:
    '''
    fw_file = open(out_file, 'w')
    for i, line in enumerate(open(filename)):
        line = line.rstrip('\r\n')
        lines = line.split('\t')
        stock_code = lines[0].strip()
        tm = lines[1].strip()
        pprice = lines[2].strip()
        high = lines[3].strip()
        low = lines[4].strip()
        close = lines[5].strip()
        volume = lines[6].strip()
        amount = lines[7].strip()
        # 时间处理成特征
        dt, year_day, year_month, month_day, week_day, hour_of_day, minute_of_hour = parse_time(tm)
        year_day_oh = to_onehot(year_day, 367)
        year_day_list = [str(ele) for ele in year_day_oh]
        year_day_str = ','.join(year_day_list)
        year_month_oh = to_onehot(year_month, 13)
        year_month_list = [str(ele) for ele in year_month_oh]
        year_month_str = ','.join(year_month_list)
        month_day_oh = to_onehot(month_day, 32)
        month_day_list = [str(ele) for ele in month_day_oh]
        month_day_str = ','.join(month_day_list)
        week_day_oh = to_onehot(week_day, 8)
        week_day_list = [str(ele) for ele in week_day_oh]
        week_day_str = ','.join(week_day_list)
        hour_of_day_oh = to_onehot(hour_of_day, 25)
        hour_of_day_list = [str(ele) for ele in hour_of_day_oh]
        hour_of_day_str = ','.join(hour_of_day_list)
        minute_of_hour_oh = to_onehot(minute_of_hour, 61)
        minute_of_hour_list = [str(ele) for ele in minute_of_hour_oh]
        minute_of_hour_str = ','.join(minute_of_hour_list)
        # stock_code 转成hash
        hash_layer = tf.keras.layers.Hashing(num_bins=10000, salt=9999)
        stock_code_hash = str(hash_layer([stock_code]).numpy()[0])
        # print(stock_code_hash, stock_code)

        # 输出数据
        row_lines = list()
        row_lines.append(dt)
        row_lines.append(pprice)
        row_lines.append(high)
        row_lines.append(low)
        row_lines.append(close)
        row_lines.append(volume)
        row_lines.append(stock_code_hash)

        row_lines.append(year_day_str)
        row_lines.append(year_month_str)
        row_lines.append(month_day_str)
        row_lines.append(week_day_str)
        row_lines.append(hour_of_day_str)
        row_lines.append(minute_of_hour_str)
        row_line = '\t'.join(row_lines)
        # print(row_line)
        fw_file.write(row_line+"\n")

def parse_time(time_str):
    ''' 根据时间获取处理后的时间特征
    :param time_str:
    :return:
    '''
    # 解析时间字符串
    timestamp = datetime.strptime(time_str, '%Y%m%d%H%M%S')
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

def gen_lookup_json():
    '''
      生成lookup数据
    :return:
    '''

    ## 时间转换
    # 一年中第几天,第几天不是交易日会有影响？
    ind = 0
    year_json = open('dataset/lookup_year.json', 'w')
    year_dict = dict()
    for i in range(365):
        ind += 1

    # 一月中第几天
    # 一个星期中第几天
    # 一天中第几个小时
    # 一小时中第几个分钟


    # 股票代码转成索引


if __name__ == '__main__':
    filename = 'dataset/hs_all_data.txt'
    out_file = 'dataset/hs_all_trainset.txt'
    gen_lookup_data(filename, out_file)
    quit()
    flag = sys.argv[1]
    if 'gen_lookup_data' == flag:
        filename = sys.argv[2]
        out_file = sys.argv[3]
        gen_lookup_data(filename, out_file)
    elif 'gen_lookup_json' == flag:
        gen_lookup_json()
    else:
        print('sys param error!')