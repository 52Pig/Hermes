import os
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
import gzip


def process_quant_data(input_dir, output_dir, stock_pool=None):
    """
    处理量化数据，为每分钟行情添加label和weight字段
    :param input_dir: 原始数据目录 (e.g. 'dataset/quant_data')
    :param output_dir: 处理后的输出目录
    :param stock_pool: 股票池列表，如果提供则只处理这些股票
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有数据文件并按日期排序
    files = sorted(f for f in os.listdir(input_dir) if f.startswith('hs_quant_base_') and f.endswith('.json'))
    print(f"Found {len(files)} data files in {input_dir}")

    # 创建日期到文件的映射
    date_to_file = {}
    for f in files:
        try:
            # 从文件名中提取日期: hs_quant_base_YYYYMMDD.json
            date_str = f.split('_')[-1].split('.')[0]
            date_to_file[date_str] = f
        except:
            print(f"Warning: Skipping invalid filename format: {f}")

    # 按日期顺序处理文件
    sorted_dates = sorted(date_to_file.keys())
    for i in range(len(sorted_dates) - 1):  # 跳过最后一天，因为没有次日数据
        current_date = sorted_dates[i]
        next_date = sorted_dates[i + 1]

        current_file = date_to_file[current_date]
        next_file = date_to_file[next_date]

        print(f"\nProcessing {current_date} -> {next_date}")
        print(f"  Current file: {current_file}")
        print(f"  Next day file: {next_file}")

        # 加载下一个交易日的数据（用于计算label）
        next_day_data = load_next_day_data(
            os.path.join(input_dir, next_file),
            stock_pool
        )

        # 处理当前交易日的数据
        process_single_day(
            input_path=os.path.join(input_dir, current_file),
            output_path=os.path.join(output_dir, current_file),
            next_day_data=next_day_data,
            current_date=current_date,
            next_date=next_date,
            stock_pool=stock_pool
        )


def load_next_day_data(file_path, stock_pool=None):
    """加载下一个交易日的数据并构建索引"""
    next_day_data = {}
    loaded_stocks = 0

    print(f"  Loading next day data: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # print(f"line={line}")
            # print(f"stock_pool={stock_pool}")
            # try:
            data = json.loads(line.strip())
            # print(data)
            # print(data.keys[0])
            stock_code = list(data.keys())[0]

            # stock_code = data['code']  # e.g. "600051.SH"

            # 如果指定了股票池且当前股票不在池中，则跳过
            if stock_pool and stock_code not in stock_pool:
                continue

            # 构建时间索引：{股票代码: {时间点: 分钟数据}}
            time_index = {}
            for minute_data in data[stock_code]['d']:
                # print(f"minute_data={minute_data}")
                # 提取时间部分 (HHMMSS)
                time_key = minute_data[0][8:]  # 从完整时间戳中取后6位
                time_index[time_key] = minute_data

            next_day_data[stock_code] = time_index
            loaded_stocks += 1

    print(f"  Loaded data for {loaded_stocks} stocks")
    return next_day_data


def get_label_by_zdf(zdf):
    """
    根据涨跌幅设计的目标
    :param zdf:
    :return:
    """
    label = -10
    if zdf <= -0.09:
        label = -9
    elif -0.09 < zdf <= -0.08:
        label = -8
    elif -0.08 < zdf <= -0.07:
        label = -7
    elif -0.07 < zdf <= -0.06:
        label = -6
    elif -0.06 < zdf <= -0.05:
        label = -5
    elif -0.05 < zdf <= -0.04:
        label = -4
    elif -0.04 < zdf <= -0.03:
        label = -3
    elif -0.03 < zdf <= -0.02:
        label = -2
    elif -0.02 < zdf <= -0.01:
        label = -1
    elif -0.01 < zdf <= 0:
        label = 0
    elif 0 < zdf <= 0.01:
        label = 1
    elif 0.01 < zdf <= 0.02:
        label = 2
    elif 0.02 < zdf <= 0.03:
        label = 3
    elif 0.03 < zdf <= 0.04:
        label = 4
    elif 0.04 < zdf <= 0.05:
        label = 5
    elif 0.05 < zdf <= 0.06:
        label = 6
    elif 0.06 < zdf <= 0.07:
        label = 7
    elif 0.07 < zdf <= 0.08:
        label = 8
    elif 0.08 < zdf <= 0.09:
        label = 9
    elif 0.09 > zdf:
        label = 10
    return label


def get_weight(zdf, current_close, next_first_open, next_last_close):
    """权重计算"""
    weight = 1.0
    open_weight = max(0.01, (0.1 + (next_first_open - current_close) / current_close) * 10.0)
    close_weight = max(0.01, (0.1 + (next_last_close - current_close) / current_close) * 100.0)

    weight = open_weight + close_weight
    return weight

def process_single_day(input_path, output_path, next_day_data, current_date, next_date, stock_pool=None):
    """处理单个交易日的数据文件"""
    processed_count = 0
    skipped_count = 0
    total_minutes = 0
    processed_minutes = 0

    with open(input_path, 'r', encoding='utf-8') as fin, \
            open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            try:
                data = json.loads(line.strip())
                stock_code = list(data.keys())[0]
                # 如果指定了股票池且当前股票不在池中，则跳过
                if stock_pool and stock_code not in stock_pool:
                    skipped_count += 1
                    continue

                # 获取股票数据的主键
                stock_data = data[stock_code]
                # 检查下一个交易日是否有该股票的数据
                if stock_code not in next_day_data:
                    skipped_count += 1
                    continue

                # 处理每分钟数据
                new_minute_data = []
                for minute_data in stock_data['d']:
                    total_minutes += 1
                    # 提取时间部分 (HHMMSS)
                    time_key = minute_data[0][8:]

                    # 检查下一个交易日是否有相同时间点的数据
                    if time_key not in next_day_data[stock_code]:
                        continue
                    # tt = next_day_data[stock_code]
                    # print(f"tt={tt}")
                    #tt={'093000': ['20240507093000', 16.0, 16.0, 16.0, 16.0, 3051, 4881600, 0.0, 0.0, 0.0, 0.0093], '093100': ['20240507093100', 15.99, 16.15, 16.16, 15.99, 14732, 23658727, 0.17, 0.9375, 0.15, 0.0448], ......, '150000': ['20240507150000', 16.09, 16.09, 16.09, 16.09, 3335, 5366015, 0.0, 0.1245, 0.02, 0.0101]}

                    next_minute = next_day_data[stock_code][time_key]

                    # 计算label：次日相同时间点的涨跌幅百分比
                    current_close = minute_data[2]  # 当前分钟收盘价
                    next_close = next_minute[2]  # 次日相同时间收盘价
                    zdf = (next_close / current_close - 1)
                    label = get_label_by_zdf(zdf)

                    # 计算weight：基于成交量的对数权重
                    #['date', 'open', 'close', 'high', 'low', 'volume', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']

                    # 次日开盘价
                    next_first_open = next_day_data[stock_code]["093000"][1]
                    next_last_close = next_day_data[stock_code]["145500"][2]

                    # 实现函数：根据label和下一天的open和close判断权重
                    weight = get_weight(zdf, current_close, next_first_open, next_last_close)


                    # 创建新的分钟数据，添加label和weight
                    new_minute = minute_data + [label, weight]
                    new_minute_data.append(new_minute)
                    processed_minutes += 1

                # 如果有处理过的分钟数据，则更新并写入
                if new_minute_data:
                    stock_data['d'] = new_minute_data
                    data[stock_code] = stock_data
                    fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                    processed_count += 1
                else:
                    skipped_count += 1

            except Exception as e:
                print(f"Error processing stock in {input_path}: {e}")
                skipped_count += 1

    print(f"  Processed: {processed_count} stocks, Skipped: {skipped_count} stocks")
    print(f"  Minutes: total={total_minutes}, processed={processed_minutes}")
    print(f"  Output saved to: {output_path}")


def calculate_label_weight_stats(file_path):
    """计算label和weight的统计信息（用于验证）"""
    label_values = []
    weight_values = []
    stock_count = 0
    minute_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                stock_key = [k for k in data.keys() if k not in ['code', 'name']][0]
                stock_data = data[stock_key]

                for minute_data in stock_data['d']:
                    # 确保数据格式正确 (应有13个字段)
                    if len(minute_data) < 13:
                        continue

                    label = minute_data[11]  # 第12个字段是label
                    weight = minute_data[12]  # 第13个字段是weight
                    label_values.append(label)
                    weight_values.append(weight)
                    minute_count += 1

                stock_count += 1
            except Exception as e:
                print(f"Error processing line: {e}")
                continue

    if not label_values:
        print("No data found for statistics")
        return

    print(f"\nStatistics for {file_path}:")
    print(f"  Stocks: {stock_count}, Minutes: {minute_count}")
    print(
        f"  Label - Min: {min(label_values):.4f}%, Max: {max(label_values):.4f}%, Avg: {sum(label_values) / len(label_values):.4f}%")
    print(
        f"  Weight - Min: {min(weight_values):.4f}, Max: {max(weight_values):.4f}, Avg: {sum(weight_values) / len(weight_values):.4f}")


# 使用示例
if __name__ == "__main__":
    config = {
        'data_dir': 'D:/tool/dataset/quant_data',
        'stock_pool_file': 'D:/tool/dataset/stock_pools.json',
        'output_dir': 'D:/tool/dataset/train_data/',
        'output_file_prefix': 'hs_quant_base_'
    }

    INPUT_DIR = config["data_dir"]
    OUTPUT_DIR = config["output_dir"]

    # 加载股票池
    stfile = config['stock_pool_file']
    sc_dict = json.load(open(stfile))
    # print(f"sc_dict={sc_dict}")
    stock_pool = list(sc_dict.keys())


    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 处理所有数据
    process_quant_data(INPUT_DIR, OUTPUT_DIR, stock_pool)

    # 验证处理后的数据（可选）
    processed_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('hs_quant_base_') and f.endswith('.json')]
    if processed_files:
        sample_file = os.path.join(OUTPUT_DIR, processed_files[0])
        print(f"\nValidating sample file: {sample_file}")
        calculate_label_weight_stats(sample_file)
    else:
        print("No processed files found for validation")