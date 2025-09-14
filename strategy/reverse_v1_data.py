import os
import pandas as pd

# 配置数据路径
source_dir = '../dataset/source_data'

# 获取最近三个交易日目录
date_dirs = [d for d in os.listdir(source_dir)
             if os.path.isdir(os.path.join(source_dir, d)) and len(d) == 8 and d.isdigit()]
date_dirs.sort(reverse=True)
recent_dates = date_dirs[:3]
if len(recent_dates) < 3:
    raise ValueError("需要至少三个交易日数据")

recent_dates_sorted = sorted(recent_dates)  # 日期升序排列
print(f"recent_dates_sorted={recent_dates_sorted}")

# 存储股票数据结构
stocks_data = {}

for date in recent_dates:
    date_path = os.path.join(source_dir, date)
    csv_files = [f for f in os.listdir(date_path) if f.endswith('.csv')]
    csv_files.sort(key=lambda x: x.split('_')[1])  # 按时间排序

    for csv_file in csv_files:
        # 获取文件中的时间
        time_str = csv_file.split('_')[1].split('.')[0]  # 格式：HH-MM
        hour, minute = map(int, time_str.split('-'))

        # 排除09:15到09:30之间的文件
        if (hour == 9 and 15 <= minute <= 30):
            continue

        file_path = os.path.join(date_path, csv_file)
        df = pd.read_csv(file_path, index_col=0, dtype={'代码':str})

        # 过滤北交所股票（代码以8/4开头）
        # print(type(df['代码'])) # == '2492':
            # print('------', df['代码'])
        df['代码'] = df['代码'].astype(str)
        df = df[~df['代码'].str.startswith(('8', '4', '92', '688'))]
        # # 去掉科创板
        # df = df[~df['代码'].str.startswith('688')]
        # # 去掉ST开头
        # df = df[(~df['代码'].str.startswith('87')) & (~df['代码'].str.startswith('83')) & (
        #     ~df['代码'].str.startswith('92'))]

        for _, row in df.iterrows():
            code = row['代码']     #.astype(str)

            if code == '2492':
                print('------', df['代码'])
            if code not in stocks_data:
                stocks_data[code] = {}
            if date not in stocks_data[code]:
                stocks_data[code][date] = {
                    'name': row['名称'],
                    'open': row['今开'],
                    'high': row['最高'],
                    'low': row['最低'],
                    'close': row['昨收'],
                    'volume': row['成交量'] if pd.notna(row['成交量']) else 0
                }
            else:
                # 更新价格信息
                current = stocks_data[code][date]
                current['high'] = max(current['high'], row['最新价'])
                current['low'] = min(current['low'], row['最新价'])
                current['close'] = row['最新价']
                if pd.notna(row['成交量']):
                    current['volume'] = row['成交量']

# 筛选符合条件的股票
selected_stocks = []
for code in stocks_data:
    stock = stocks_data[code]

    # 检查是否包含全部三日数据
    if not all(date in stock for date in recent_dates_sorted):
        continue

    # 获取三日收盘价（按日期升序）
    closes = [stock[date]['close'] for date in recent_dates_sorted]

    # 检查下降趋势
    if not (closes[0] > closes[1] > closes[2]):
        continue

    # 检查无上影线
    valid = True
    for date in recent_dates_sorted:
        data = stock[date]
        max_oc = max(data['open'], data['close'])
        if data['high'] != max_oc:
            valid = False
            break
    if not valid:
        continue

    # 收集结果
    selected_stocks.append({
        'code': code,
        'name': stock[recent_dates_sorted[-1]]['name'],
        'volumes': {date: stock[date]['volume'] for date in recent_dates_sorted}
    })

# 输出结果
print("符合条件的股票：")
import json
import datetime

formatted_datetime = datetime.datetime.now().strftime("%Y%m%d")
out_file = '../dataset/reverse_data/pools_data.' + formatted_datetime
print("输入文件：", out_file)
fw_file = open(out_file, 'w')
for stock in selected_stocks:
    fw_file.write(json.dumps(stock)+"\n")

#     print(f"股票代码：{stock['code']} 名称：{stock['name']}")
#     print("最近三日成交量：")
#     for date in recent_dates_sorted:
#         print(f"{date}: {stock['volumes'][date]}")
#     print("\n")