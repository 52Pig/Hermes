import pandas as pd
import numpy as np
from datetime import timedelta

# 设置随机数种子以获得可重复的结果
np.random.seed(42)

# 生成日期范围
start_date = pd.to_datetime('2019-01-01')
end_date = pd.to_datetime('2019-12-31')
date_range = pd.date_range(start_date, end_date, freq='B')  # 仅工作日

# 生成模拟股票数据
data = {
    'Date': date_range,
    'Open': np.random.uniform(100, 200, size=len(date_range)),
    'High': np.random.uniform(105, 205, size=len(date_range)),
    'Low': np.random.uniform(95, 195, size=len(date_range)),
    'Close': np.random.uniform(102, 202, size=len(date_range)),
    'Volume': np.random.randint(100000, 500000, size=len(date_range))
}

# 创建DataFrame
df = pd.DataFrame(data)

# 将日期列设置为索引
df.set_index('Date', inplace=True)

# 保存到CSV文件
df.to_csv('dataset/btdemo.csv', columns=['Open', 'High', 'Low', 'Close', 'Volume'])

print("CSV file has been created with 1000 rows of simulated stock data.")