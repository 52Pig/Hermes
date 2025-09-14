import os
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
import gzip
import hashlib

import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import StandardScaler


def technical_normalization(df, features):
    """
    完整的技术指标归一化流程
    包含: ATR归一化、布林带位置、RSI、MACD等

    参数:
    df: 包含 OHLCV 列的 DataFrame
    features: 要使用的特征列表 ['atr_norm', 'bb_position', 'rsi', ...]

    返回:
    包含所有归一化特征的 DataFrame
    """
    df = df.copy()
    results = pd.DataFrame(index=df.index)

    # 1. 基于ATR的归一化 (核心价格特征)
    if 'atr_norm' in features:
        df['MA20'] = df['close'].rolling(20).mean()
        df['ATR14'] = talib.ATR(df['high'], df['low'], df['close'], 14)
        results['atr_norm'] = (df['close'] - df['MA20']) / df['ATR14']

    # 2. 布林带位置归一化
    if 'bb_position' in features:
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        results['bb_position'] = (df['close'] - lower) / (upper - lower)
        results['bb_position'] = results['bb_position'].clip(0, 1)

    # 3. RSI (相对强弱指数)
    if 'rsi' in features:
        results['rsi'] = talib.RSI(df['close'], timeperiod=14)
        # RSI 归一化到 [0,1]
        results['rsi'] = (results['rsi'] - 30) / (70 - 30)  # 典型超买超卖区间
        results['rsi'] = results['rsi'].clip(0, 1)

    # 4. MACD (指数平滑异同平均线)
    if 'macd' in features:
        macd, signal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        results['macd_diff'] = macd - signal  # MACD 柱状图值
        # 使用Z-Score标准化
        scaler = StandardScaler()
        results['macd_diff'] = scaler.fit_transform(results[['macd_diff']].fillna(0))

    # 5. 成交量标准化
    if 'volume_norm' in features:
        # 使用ATR缩放成交量
        df['ATR14'] = df['ATR14'].fillna(method='bfill')
        results['volume_norm'] = df['volume'] / df['ATR14']
        # 去除极端值
        results['volume_norm'] = np.clip(results['volume_norm'],
                                         results['volume_norm'].quantile(0.01),
                                         results['volume_norm'].quantile(0.99))

    # 6. 蜡烛图相对位置特征
    if 'candle_features' in features:
        # 开盘价相对于昨日收盘的位置
        results['open_rel'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        # 当日振幅标准化
        daily_range = df['high'] - df['low']
        results['close_pos'] = (df['close'] - df['low']) / daily_range.replace(0, 1e-5)

        # 实体大小
        body_size = abs(df['close'] - df['open'])
        results['body_ratio'] = body_size / daily_range.replace(0, 1e-5)

    # 处理NaN值
    results = results.fillna(method='bfill').fillna(0)

    return results


# 使用示例
# features = ['atr_norm', 'bb_position', 'rsi', 'volume_norm']
# normalized_features = technical_normalization(ohlc_data, features)

def process_quant_data(input_dir, output_dir, stock_pool=None, stc_dict=None):
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
            output_path=os.path.join(output_dir, f"{current_date}_clean.txt"),
            next_day_data=next_day_data,
            current_date=current_date,
            next_date=next_date,
            stock_pool=stock_pool,
            stc_dict=stc_dict
        )


def load_next_day_data(file_path, stock_pool=None):
    """加载下一个交易日的数据并构建索引"""
    next_day_data = {}
    loaded_stocks = 0

    print(f"  Loading next day data: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            stock_code = list(data.keys())[0]

            # 如果指定了股票池且当前股票不在池中，则跳过
            if stock_pool and stock_code not in stock_pool:
                continue

            # 构建时间索引：{股票代码: {时间点: 分钟数据}}
            time_index = {}
            for minute_data in data[stock_code]['d']:
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
    label = 0
    if zdf <= -0.09:
        label = 1
    elif -0.09 < zdf <= -0.08:
        label = 2
    elif -0.08 < zdf <= -0.07:
        label = 3
    elif -0.07 < zdf <= -0.06:
        label = 4
    elif -0.06 < zdf <= -0.05:
        label = 5
    elif -0.05 < zdf <= -0.04:
        label = 6
    elif -0.04 < zdf <= -0.03:
        label = 7
    elif -0.03 < zdf <= -0.02:
        label = 8
    elif -0.02 < zdf <= -0.01:
        label = 9
    elif -0.01 < zdf <= 0:
        label = 10
    elif 0 < zdf <= 0.01:
        label = 11
    elif 0.01 < zdf <= 0.02:
        label = 12
    elif 0.02 < zdf <= 0.03:
        label = 13
    elif 0.03 < zdf <= 0.04:
        label = 14
    elif 0.04 < zdf <= 0.05:
        label = 15
    elif 0.05 < zdf <= 0.06:
        label = 16
    elif 0.06 < zdf <= 0.07:
        label = 17
    elif 0.07 < zdf <= 0.08:
        label = 18
    elif 0.08 < zdf <= 0.09:
        label = 19
    elif zdf > 0.09:
        label = 20
    return label


def get_weight(zdf, current_close, next_first_open, next_last_close):
    """权重计算"""
    open_weight = max(0.01, (0.1 + (next_first_open - current_close) / current_close) * 10.0)
    close_weight = max(0.01, (0.1 + (next_last_close - current_close) / current_close) * 100.0)
    weight = open_weight + close_weight
    return weight


def process_single_day(input_path, output_path, next_day_data, current_date, next_date, stock_pool=None, stc_dict=None):
    """处理单个交易日的数据文件，输出TXT格式"""
    processed_count = 0
    skipped_count = 0
    total_minutes = 0
    processed_minutes = 0

    with open(input_path, 'r', encoding='utf-8') as fin, \
            open(output_path, 'w', encoding='utf-8') as fout:

        # 写入表头
        header = "code\tlabel\tweight\topen\tclose\thigh\tlow\tvolume\tamount\tamplitude\tchange_pct\tchange_amount\tturnover_rate\ttime"
        fout.write(header + "\n")

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
                stock_hash = stc_dict.get(stock_data['code'], "-1")

                # 处理每分钟数据
                for minute_data in stock_data['d']:
                    total_minutes += 1
                    # 提取时间部分 (HHMMSS)
                    time_key = minute_data[0][8:]

                    # 检查下一个交易日是否有相同时间点的数据
                    if time_key not in next_day_data[stock_code]:
                        continue

                    next_minute = next_day_data[stock_code][time_key]

                    # 计算label：次日相同时间点的涨跌幅百分比
                    current_close = minute_data[2]  # 当前分钟收盘价
                    next_close = next_minute[2]  # 次日相同时间收盘价
                    zdf = (next_close / current_close - 1)
                    label = get_label_by_zdf(zdf)

                    # 计算weight：基于成交量的对数权重
                    # 次日开盘价
                    next_first_open = next_day_data[stock_code]["093000"][1]
                    next_last_close = next_day_data[stock_code]["145500"][2]
                    weight = get_weight(zdf, current_close, next_first_open, next_last_close)

                    # 准备输出字段
                    # 原始分钟数据字段: [时间, 开盘价, 收盘价, 最高价, 最低价, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率]
                    output_fields = [
                        stock_hash,  # code
                        str(label),  # label
                        f"{weight:.6f}",  # weight
                        f"{minute_data[1]:.4f}",  # open
                        f"{minute_data[2]:.4f}",  # close
                        f"{minute_data[3]:.4f}",  # high
                        f"{minute_data[4]:.4f}",  # low
                        str(minute_data[5]),  # volume
                        f"{minute_data[6]:.4f}",  # amount (成交额)
                        f"{minute_data[7]:.4f}",  # amplitude (振幅)
                        f"{minute_data[8]:.4f}",  # change_pct (涨跌幅)
                        f"{minute_data[9]:.4f}",  # change_amount (涨跌额)
                        f"{minute_data[10]:.6f}",  # turnover_rate (换手率)
                        minute_data[0]  # 完整时间戳
                    ]

                    # 写入一行数据
                    fout.write("\t".join(output_fields) + "\n")
                    processed_minutes += 1
                    processed_count += 1

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
        # 跳过表头
        next(f)
        for line in f:
            try:
                fields = line.strip().split('\t')
                if len(fields) < 14:
                    continue

                label = int(fields[1])
                weight = float(fields[2])

                label_values.append(label)
                weight_values.append(weight)
                minute_count += 1
            except Exception as e:
                print(f"Error processing line: {e}")
                continue

    if not label_values:
        print("No data found for statistics")
        return

    print(f"\nStatistics for {file_path}:")
    print(f"  Minutes: {minute_count}")
    print(
        f"  Label - Min: {min(label_values)}, Max: {max(label_values)}, Avg: {sum(label_values) / len(label_values):.2f}")
    print(
        f"  Weight - Min: {min(weight_values):.4f}, Max: {max(weight_values):.4f}, Avg: {sum(weight_values) / len(weight_values):.4f}")


def gen_lookup_json(sc_dict, lookup_code_file):
    """
    生成 lookup 数据（PyTorch 版本）
    :return:
    """
    salt = str(9999)          # 对应原 salt
    num_bins = 100000          # 对应原 num_bins

    stc_dict = {}

    for code, info in sc_dict.items():
        stock = info.get("code", "")
        if stock == "":
            continue
        # 等价于 TF 的 Hashing(num_bins=10000, salt=9999)
        salted = salt + stock                         # 加盐
        hash_val = int(hashlib.md5(salted.encode()).hexdigest(), 16)
        sc_hash = str(hash_val % num_bins)            # 取模映射到 [0, num_bins-1]
        stc_dict[stock] = sc_hash

    with open(lookup_code_file, 'w', encoding='utf-8') as f:
        json.dump(stc_dict, f, indent=4, ensure_ascii=False)
    return stc_dict

if __name__ == "__main__":
    config = {
        'data_dir': 'D:/tool/dataset/quant_data',
        'stock_pool_file': 'D:/tool/dataset/stock_pools.json',
        'output_dir': 'D:/tool/dataset/train_data/',
        'lookup_code_file': "D:/tool/dataset/lookup_stock_code.json"
    }

    INPUT_DIR = config["data_dir"]
    OUTPUT_DIR = config["output_dir"]

    # 加载股票池
    stfile = config['stock_pool_file']
    sc_dict = json.load(open(stfile))
    stock_pool = list(sc_dict.keys())

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    ## 生成股票hash编码
    stc_dict = gen_lookup_json(sc_dict, config['lookup_code_file'])

    # 处理所有数据
    process_quant_data(INPUT_DIR, OUTPUT_DIR, stock_pool, stc_dict)

    # 验证处理后的数据（可选）
    processed_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')]
    if processed_files:
        sample_file = os.path.join(OUTPUT_DIR, processed_files[0])
        print(f"\nValidating sample file: {sample_file}")
        calculate_label_weight_stats(sample_file)
    else:
        print("No processed files found for validation")