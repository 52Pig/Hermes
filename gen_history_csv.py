import pandas as pd
import numpy as np
import datetime

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 生成日期范围
start_date = datetime.date(2010, 1, 1)
end_date = datetime.date(2023, 1, 1)
date_range = pd.date_range(start_date, end_date, freq='B')  # 'B' 表示工作日

# 生成至少 1000 行数据
num_rows = 1000
dates = np.random.choice(date_range, num_rows, replace=True)
dates = np.sort(dates)  # 确保日期是升序的

# 生成开盘价、最高价、最低价和收盘价
opens = np.random.uniform(low=50, high=150, size=num_rows)
highs = opens + np.random.uniform(low=0.5, high=10, size=num_rows)
lows = opens - np.random.uniform(low=0.5, high=10, size=num_rows)
closes = np.random.uniform(low=lows, high=highs)

# 创建 DataFrame
data = pd.DataFrame({
    'Date': dates,
    'Open': opens,
    'High': highs,
    'Low': lows,
    'Close': closes
})

# 处理日期
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values(by='Date', inplace=True)

# 保存到 CSV 文件
data.to_csv('historical_data.csv', index=False)

print("historical_data.csv 文件已生成")
