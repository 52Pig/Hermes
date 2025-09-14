import os, sys
import json
import datetime
import pandas as pd
import numpy as np
sys.path.append('../')
import json
import time
from utils import utils

#
# formatted_datetime = datetime.datetime.now().strftime("%Y%m%d")
# trade_days = utils.get_a_share_trade_days(start_date=formatted_datetime)
#
# if len(trade_days) == 0:
#     print(f"[DEBUG]{formatted_datetime} is not a valid trade day! program quit!")
#     quit()

# ed = trade_days[-1]


# 配置参数
config = {
    'data_dir': 'D:/tools/dataset/quant_data',
    'stock_pool_file': 'D:/tools/dataset/stock_pools.json',
    'output_dir': 'D:/tools/dataset/strategy_data/',
    'output_file_prefix': 'reverse_moving_average_bull_track_',
    'ma_windows': [5, 10, 20, 30],
    'min_days': 40,  # 计算均线所需的最小天数
    'history_days': 254,  # 需要加载的历史交易日天数（用于判断涨停等）
    'end_date': '20250613'
    # 'end_date': ed
}

# 确保输出目录存在
os.makedirs(config['output_dir'], exist_ok=True)


def load_stock_pool(file_path):
    """加载股票池数据"""
    return json.load(open(file_path))


def get_trading_dates(data_dir):
    """获取交易日期列表并按日期排序"""
    files = [f for f in os.listdir(data_dir)
             if f.startswith('hs_quant_base_') and f.endswith('.json')]
    dates = [f.split('_')[3].split('.')[0] for f in files]
    # 过滤掉超过end_date的日期
    filtered_dates = [d for d in dates if d <= config['end_date']]
    sorted_dates = sorted(filtered_dates, key=lambda x: datetime.datetime.strptime(x, '%Y%m%d'))
    return sorted_dates


def load_stock_data(data_dir, target_codes, trading_dates):
    """加载股票数据并构建时间序列"""
    # 初始化数据结构
    stock_close_prices = {code: [] for code in target_codes}
    stock_info = {code: {} for code in target_codes}

    total_files = len(trading_dates)
    print(f"\n开始加载股票数据，共需处理 {total_files} 个交易日文件...")

    # 按日期顺序处理文件
    for i, date_str in enumerate(trading_dates):
        file_path = os.path.join(data_dir, f'hs_quant_base_{date_str}.json')
        if not os.path.exists(file_path):
            continue
        # 显示进度
        progress = i / total_files * 100
        sys.stdout.write(f"\r处理进度: {i}/{total_files} ({progress:.1f}%) - 当前日期: {date_str}")
        sys.stdout.flush()

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                for code, stock_data in data.items():
                    if code not in target_codes:
                        continue
                    # 获取当天最后一条记录的收盘价
                    if 'd' in stock_data and stock_data['d']:
                        last_record = stock_data['d'][-1]
                        close_price = last_record[4]  # 收盘价在索引位置4
                        stock_close_prices[code].append(close_price)

                    # 保存最新股票信息
                    if 'i' in stock_data:
                        info = stock_data['i']
                        stock_info[code] = {
                            'code': stock_data.get('code', ''),
                            'name': stock_data.get('name', ''),
                            'total_mv': info[0] if len(info) > 0 else -1,
                            'circ_mv': info[1] if len(info) > 1 else -1,
                            'industry': info[2] if len(info) > 2 else '',
                            'industry_id': info[3] if len(info) > 3 else '',
                            'pe_ttm': info[4] if len(info) > 4 else -1,
                            'pb': info[5] if len(info) > 5 else -1
                        }

    return stock_close_prices, stock_info


def calculate_technical_indicators(stock_data, stock_info):
    """计算技术指标"""
    results = {}
    total_stocks = len(stock_data)
    processed = 0
    print(f"\n开始计算技术指标，共需处理 {total_stocks} 只股票...")

    for code, prices in stock_data.items():
        processed += 1
        progress = processed / total_stocks * 100
        sys.stdout.write(f"\r处理进度: {processed}/{total_stocks} ({progress:.1f}%) - 当前股票: {code}")
        sys.stdout.flush()

        if len(prices) < config['min_days']:
            # 数据不足
            results[code] = {
                **stock_info.get(code, {}),
                'is_target': '0',
                'error': '数据不足'
            }
            continue

        try:
            # 创建DataFrame
            df = pd.DataFrame(prices, columns=['close'])

            # 计算均线 - 只使用最后min_days天的数据计算
            recent_prices = df['close'].tail(config['min_days'])
            for window in config['ma_windows']:
                df[f'MA{window}'] = np.nan
                df[f'MA{window}'].iloc[-config['min_days']:] = recent_prices.rolling(window=window).mean()

            # 使用最后有效数据
            valid_df = df.dropna()
            if valid_df.empty:
                results[code] = {
                    **stock_info.get(code, {}),
                    'is_target': '0',
                    'error': '计算均线失败'
                }
                continue

            # 获取指标值
            last_row = valid_df.iloc[-1]
            ma5 = last_row['MA5']
            ma10 = last_row['MA10']
            ma20 = last_row['MA20']
            ma30 = last_row['MA30']

            # 最近7天价格波动
            last_7_close = df['close'].tail(7)
            last_7_min = last_7_close.min()
            last_7_max = last_7_close.max()
            last_7_diff = (last_7_max - last_7_min) / last_7_min if last_7_min > 0 else 0

            # 最近两天涨跌幅
            last_closes = df['close'].tail(10).values
            if len(last_closes) >= 3:
                last_1d_zf = (last_closes[-1] - last_closes[-2]) / last_closes[-2]
                last_2d_zf = (last_closes[-2] - last_closes[-3]) / last_closes[-3]
                last_3d_zf = (last_closes[-3] - last_closes[-4]) / last_closes[-4]
                last_4d_zf = (last_closes[-4] - last_closes[-5]) / last_closes[-5]
                last_5d_zf = (last_closes[-5] - last_closes[-6]) / last_closes[-6]
            else:
                last_1d_zf = 0
                last_2d_zf = 0
                last_3d_zf = 0
                last_4d_zf = 0
                last_5d_zf = 0

            # 涨停检测 - 使用全部历史数据
            up_limit_days = 0
            is_up_limit_half_year = False
            half_year_days = min(config['history_days'], len(df))  # 使用配置的历史天数

            # 检测最近半年涨停
            for i in range(1, half_year_days):
                if i >= len(df):
                    break
                prev_close = df['close'].iloc[-i - 1]
                cur_close = df['close'].iloc[-i]
                if abs(cur_close - round(prev_close * 1.1, 2)) < 1e-6:
                    is_up_limit_half_year = True
                    break

            # 检测最近5天涨停
            for i in range(1, min(6, len(df))):
                if i >= len(df):
                    break
                prev_close = df['close'].iloc[-i - 1]
                cur_close = df['close'].iloc[-i]
                if abs(cur_close - round(prev_close * 1.1, 2)) < 1e-6:
                    up_limit_days += 1

            # 均线条件判断
            is_ma5_lowest = (ma10 >= ma5 and ma20 >= ma5 and ma30 >= ma5)
            is_not_up_limit = (up_limit_days == 0)

            # 目标股判断条件
            latest_price = df['close'].iloc[-1]
            is_target = (
                    is_ma5_lowest and
                    is_not_up_limit and
                    is_up_limit_half_year and
                    latest_price >= 2.0 and
                    # latest_price <= 20.0 and
                    last_1d_zf <= 0.06 and
                    last_2d_zf <= 0.06 and
                    last_3d_zf <= 0.096 and
                    last_4d_zf <= 0.096 and
                    last_5d_zf <= 0.096
            )

            # 保存结果
            results[code] = {
                **stock_info.get(code, {}),
                'is_target': '1' if is_target else '0',
                'last_close': ','.join(map(str, df['close'].tail(100).tolist())),
                'MA5': ma5,
                'MA10': ma10,
                'MA20': ma20,
                'MA30': ma30,
                'up_limit_days': up_limit_days,
                'last_5d_diff': last_7_diff,
                'is_ma5_lowest': int(is_ma5_lowest),
                'is_not_up_limit': int(is_not_up_limit),
                'is_up_limit_half_year': int(is_up_limit_half_year),
                'last_1d_zf': last_1d_zf,
                'last_2d_zf': last_2d_zf
            }

        except Exception as e:
            results[code] = {
                **stock_info.get(code, {}),
                'is_target': '0',
                'error': str(e)
            }
    print("\n技术指标计算完成!")
    return results


def main():
    start_time = datetime.datetime.now()
    print(f"开始执行策略计算，当前时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置参数: {config}")

    # 1. 加载股票池
    print("\n=== 步骤1: 加载股票池 ===")
    stock_pool = load_stock_pool(config['stock_pool_file'])
    target_codes = list(stock_pool.keys())
    print(f"加载完成，共 {len(target_codes)} 只股票")

    # 2. 获取交易日期
    print("\n=== 步骤2: 获取交易日期 ===")
    trading_dates = get_trading_dates(config['data_dir'])
    if len(trading_dates) < config['min_days']:
        print(f"错误: 可用交易日不足 {config['min_days']} 天")
        return

    # 加载足够的历史数据（至少history_days天，但不超过可用数据）
    load_days = min(config['history_days'], len(trading_dates))
    print(f"获取到 {len(trading_dates)} 个交易日，将使用最后 {load_days} 个交易日")

    # 3. 加载股票数据
    print("\n=== 步骤3: 加载股票数据 ===")
    stock_close_prices, stock_info = load_stock_data(
        config['data_dir'],
        target_codes,
        trading_dates[-load_days:]  # 使用足够的历史数据
    )

    # 4. 计算技术指标
    print("\n=== 步骤4: 计算技术指标 ===")
    results = calculate_technical_indicators(stock_close_prices, stock_info)

    # 5. 添加股票池原始信息
    print("\n=== 步骤5: 合并股票池信息 ===")
    for code, data in results.items():
        if code in stock_pool:
            data.update({
                'FloatVolume': stock_pool[code].get('FloatVolume', 0),
                'TotalVolume': stock_pool[code].get('TotalVolume', 0)
            })

    # 6. 保存结果
    print("\n=== 步骤6: 保存结果 ===")
    output_file = os.path.join(
        config['output_dir'],
        f"{config['output_file_prefix']}{config['end_date']}.json"
    )



    stc_str = json.dumps(results, ensure_ascii=False, indent=4)
    fw = open(output_file, 'w')
    fw.write(stc_str+"\n")
    fw.close()

    # 7. 保存元数据
    target_count = sum(1 for data in results.values() if data.get('is_target') == '1')
    meta_data = {
        'pools_size': len(results),
        'target_num': target_count,
        'processing_time': str(datetime.datetime.now() - start_time),
        'processed_date': datetime.datetime.now().strftime("%Y%m%d"),
        'history_days_used': load_days
    }

    meta_file = output_file + '.meta'
    meta_fw = open(meta_file, 'w')
    meta_str = json.dumps(meta_data, ensure_ascii=False, indent=4)
    meta_fw.write(meta_str+"\n")
    # with open(meta_file, 'w', encoding='utf-8') as f:
    #     json.dump(meta_data, f, ensure_ascii=False, indent=2)
    meta_fw.close()
    print(f"\n处理完成! 共处理 {len(results)} 只股票，目标股 {target_count} 只")
    print(f"结果保存至: {output_file}")
    print(f"总耗时: {datetime.datetime.now() - start_time}")


if __name__ == '__main__':
    main()