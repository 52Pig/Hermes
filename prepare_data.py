
import os
import json
import glob
import concurrent
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor


class PrepareTrainData(object):
    """准备训练数据"""
    def __init__(self):
        self.lookup_tables = {}
        self.lookup_tables['stock_code'] = self.load_lookup_table("dataset/lookup_stock_code.json")

    def load_lookup_table(self, filename):
        with open(filename, "r") as f:
            return json.load(f)

    def to_onehot(self, value, num_classes):
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

    def parse_time(self, time_str):
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

    def do(self, input_parent_path, output_parent_path, cur_date):
        output_path = f"{output_parent_path}/{cur_date}_clean.txt"
        f_w = open(output_path, "w")
        input_paths = glob.glob(f"{input_parent_path}/{cur_date}/hs_all_data.txt")
        for input_path in input_paths:
            with open(input_path, "r") as f:
                for line in f:
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
                    dt, year_day, year_month, month_day, week_day, hour_of_day, minute_of_hour = self.parse_time(tm)
                    year_day_oh = self.to_onehot(year_day, 367)
                    year_day_list = [str(ele) for ele in year_day_oh]
                    year_day_str = ','.join(year_day_list)
                    year_month_oh = self.to_onehot(year_month, 13)
                    year_month_list = [str(ele) for ele in year_month_oh]
                    year_month_str = ','.join(year_month_list)
                    month_day_oh = self.to_onehot(month_day, 32)
                    month_day_list = [str(ele) for ele in month_day_oh]
                    month_day_str = ','.join(month_day_list)
                    week_day_oh = self.to_onehot(week_day, 8)
                    week_day_list = [str(ele) for ele in week_day_oh]
                    week_day_str = ','.join(week_day_list)
                    hour_of_day_oh = self.to_onehot(hour_of_day, 25)
                    hour_of_day_list = [str(ele) for ele in hour_of_day_oh]
                    hour_of_day_str = ','.join(hour_of_day_list)
                    minute_of_hour_oh = self.to_onehot(minute_of_hour, 61)
                    minute_of_hour_list = [str(ele) for ele in minute_of_hour_oh]
                    minute_of_hour_str = ','.join(minute_of_hour_list)
                    # stock_code 转成hash
                    stock_code_hash = self.lookup_tables['stock_code'].get(stock_code, '0')
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
                    f_w.write(row_line + "\n")

                    # break
        f_w.close()


if __name__ == '__main__':
    input_path = "dataset/source_data/"
    output_path = "dataset/clean_sample/"
    pre_data = PrepareTrainData()
    dates = [(input_path, output_path, i) for i in os.listdir(input_path)]
    # pre_data.do(input_path, output_path, '20240506')
    # quit()
    res_lst = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        res_dict = {pool.submit(pre_data.do, d[0], d[1], d[2]): d for d in dates}
        for future in concurrent.futures.as_completed(res_dict):
            # param = recall_res_dict[future]
            data = future.result()
            res_lst.append(data)