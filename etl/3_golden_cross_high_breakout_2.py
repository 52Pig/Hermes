import os, sys
import json
import datetime
import pandas as pd
import numpy as np

sys.path.append('../')
import json
import time
from utils import utils

formatted_datetime = datetime.datetime.now().strftime("%Y%m%d")
trade_days = utils.get_a_share_trade_days(start_date=formatted_datetime)

# 配置参数
config = {
    'data_dir': 'D:/tool/dataset/quant_data',
    'stock_pool_file': 'D:/tool/dataset/stock_pools.json',
    'output_dir': 'D:/tool/dataset/strategy_data/',
    'output_file_prefix': 'golden_cross_breakout_',
    'ma_windows': [5, 10, 20, 30, 60],
    'min_days': 70,  # 计算均线所需的最小天数
    'history_days': 200,  # 需要加载的历史交易日天数
    'lookback_period': 66,  # 3个月约66个交易日
    'end_date': '20250909',
    'tolerance_ratio': 0.98  # 接近最高点的容忍比例，98%表示允许2%的回落
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
                        open_price = stock_data['d'][1][1]
                        high_price = max(minute_data[3] for minute_data in stock_data['d'])
                        # 总成交量
                        volume = sum(minute_data[5] for minute_data in stock_data['d'][:-10])
                        stock_close_prices[code].append({
                            'date': date_str,
                            'open': open_price,
                            'close': close_price,
                            'volume': volume,
                            'high': high_price
                        })

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


def convert_to_serializable(obj):
    """将对象转换为JSON可序列化的格式"""
    if isinstance(obj, (pd.Timestamp, datetime.datetime)):
        return obj.strftime('%Y%m%d')
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


def find_first_peak_after_golden_cross(df, golden_cross_index):
    """
    在金叉后寻找第一个峰值（高点后开始下跌）
    返回峰值价格、峰值日期和是否找到有效峰值
    """
    if golden_cross_index >= len(df) - 1:
        return None, None, False

    # 从金叉点开始寻找
    start_idx = golden_cross_index
    peak_price = 0
    peak_date = None
    peak_index = -1

    # 寻找第一个明显的峰值
    for i in range(start_idx + 1, len(df) - 1):
        current_high = df.iloc[i]['high']
        next_high = df.iloc[i + 1]['high']

        # 如果当前是高点，且后一天开始下跌，认为是峰值
        if current_high > peak_price and next_high < current_high:
            peak_price = current_high
            peak_date = df.iloc[i]['date']
            peak_index = i
            break
        elif current_high > peak_price:
            # 记录当前最高点，但继续寻找
            peak_price = current_high
            peak_date = df.iloc[i]['date']
            peak_index = i

    # 如果找到了峰值，返回结果
    if peak_index > 0 and peak_price > 0:
        return peak_price, peak_date, True

    return None, None, False


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
                'error': f'数据不足，只有{len(prices)}天数据，需要{config["min_days"]}天'
            }
            continue

        try:
            # 创建DataFrame
            df = pd.DataFrame(prices)
            # 确保数据按日期排序
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df = df.sort_values('date').reset_index(drop=True)

            # 计算均线
            close_prices = df['close']
            for window in config['ma_windows']:
                df[f'MA{window}'] = close_prices.rolling(window=window).mean()

            # 移除包含NaN的行（均线计算初期的数据）
            valid_df = df.dropna().reset_index(drop=True)
            if len(valid_df) < config['lookback_period']:
                results[code] = {
                    **stock_info.get(code, {}),
                    'is_target': '0',
                    'error': f'有效数据不足，只有{len(valid_df)}天有效数据，需要{config["lookback_period"]}天'
                }
                continue

            # 策略核心逻辑
            is_target = False
            strategy_details = {
                'golden_cross_found': False,
                'first_peak_price': 0,
                'first_peak_date': None,
                'current_near_peak': False,
                'golden_cross_date': None,
                'current_to_peak_ratio': 0
            }

            # 1. 最近3个月内寻找MA5上穿MA10的金叉
            golden_cross_index = -1
            start_idx = max(0, len(valid_df) - config['lookback_period'])

            for i in range(start_idx + 1, len(valid_df)):
                # 检查金叉条件：MA5上穿MA10
                ma5_current = valid_df.iloc[i]['MA5']
                ma10_current = valid_df.iloc[i]['MA10']
                ma5_prev = valid_df.iloc[i - 1]['MA5']
                ma10_prev = valid_df.iloc[i - 1]['MA10']

                if ma5_current > ma10_current and ma5_prev <= ma10_prev:
                    golden_cross_index = i
                    strategy_details['golden_cross_found'] = True
                    strategy_details['golden_cross_date'] = valid_df.iloc[i]['date'].strftime('%Y%m%d')
                    break

            # 2. 如果找到金叉，寻找金叉后的第一个峰值
            if strategy_details['golden_cross_found'] and golden_cross_index > 0:
                peak_price, peak_date, found_peak = find_first_peak_after_golden_cross(valid_df, golden_cross_index)

                if found_peak:
                    strategy_details['first_peak_price'] = float(peak_price)
                    strategy_details['first_peak_date'] = peak_date.strftime('%Y%m%d')

                    # 3. 检查当前价格是否接近峰值（在容忍范围内）
                    current_price = valid_df.iloc[-1]['close']
                    ratio = current_price / peak_price
                    strategy_details['current_to_peak_ratio'] = float(ratio)

                    # 接近条件：当前价格在峰值价格的容忍比例范围内
                    if ratio >= config['tolerance_ratio'] and ratio <= 1.0:
                        strategy_details['current_near_peak'] = True
                        is_target = True

            # 保存结果
            current_price = float(valid_df.iloc[-1]['close'])
            ma5 = float(valid_df.iloc[-1]['MA5'])
            ma10 = float(valid_df.iloc[-1]['MA10'])

            # 转换日期为字符串格式
            last_30_dates = [d.strftime('%Y%m%d') for d in valid_df['date'].tail(30).tolist()]
            last_30_closes = [float(round(c, 4)) for c in valid_df['close'].tail(30).tolist()]

            results[code] = {
                **stock_info.get(code, {}),
                'is_target': '1' if is_target else '0',
                'strategy_signal': '接近金叉后首峰值' if is_target else '无信号',
                'current_price': current_price,
                'MA5': ma5,
                'MA10': ma10,
                'golden_cross_found': strategy_details['golden_cross_found'],
                'first_peak_price': strategy_details['first_peak_price'],
                'first_peak_date': strategy_details['first_peak_date'],
                'current_near_peak': strategy_details['current_near_peak'],
                'golden_cross_date': strategy_details['golden_cross_date'],
                'current_to_peak_ratio': strategy_details['current_to_peak_ratio'],
                'last_30_dates': ','.join(last_30_dates),
                'last_30_closes': ','.join([str(c) for c in last_30_closes])
            }

            # 输出目标股票详情
            if is_target:
                print(f"\n[目标股] {code} - 峰值价格: {strategy_details['first_peak_price']:.2f}, "
                      f"当前价: {current_price:.2f}, 比例: {strategy_details['current_to_peak_ratio']:.3f}, "
                      f"金叉日期: {strategy_details['golden_cross_date']}, 峰值日期: {strategy_details['first_peak_date']}")

        except Exception as e:
            print(f"\n处理股票 {code} 时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            results[code] = {
                **stock_info.get(code, {}),
                'is_target': '0',
                'error': f'处理错误: {str(e)}'
            }
            continue

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

    # 确保所有数据都是JSON可序列化的
    serializable_results = {}
    for code, data in results.items():
        serializable_data = {}
        for key, value in data.items():
            serializable_data[key] = convert_to_serializable(value)
        serializable_results[code] = serializable_data

    stc_str = json.dumps(serializable_results, ensure_ascii=False, indent=4)
    fw = open(output_file, 'w', encoding='utf-8')
    fw.write(stc_str + "\n")
    fw.close()

    # 7. 保存元数据
    target_count = sum(1 for data in results.values() if data.get('is_target') == '1')
    meta_data = {
        'pools_size': len(results),
        'target_num': target_count,
        'processing_time': str(datetime.datetime.now() - start_time),
        'processed_date': datetime.datetime.now().strftime("%Y%m%d"),
        'history_days_used': load_days,
        'tolerance_ratio': config['tolerance_ratio']
    }

    meta_file = output_file + '.meta'
    meta_fw = open(meta_file, 'w', encoding='utf-8')
    meta_str = json.dumps(meta_data, ensure_ascii=False, indent=4)
    meta_fw.write(meta_str + "\n")
    meta_fw.close()

    print(f"\n处理完成! 共处理 {len(results)} 只股票，目标股 {target_count} 只")
    print(f"结果保存至: {output_file}")
    print(f"总耗时: {datetime.datetime.now() - start_time}")


if __name__ == '__main__':
    main()
