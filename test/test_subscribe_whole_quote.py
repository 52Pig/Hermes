from xtquant import xtdata
from datetime import datetime
import time
import pandas as pd

code = '000560.SZ'
# 初始化一个空的 DataFrame
df_columns = ['code', 'time', 'open', 'close', 'high', 'low', 'volume']
df = pd.DataFrame(columns=df_columns)

def on_data (datas):
    global df  # 使用全局变量以便更新 DataFrame
    tick_time = datas[code]['time']
    timestamp_seconds = tick_time / 1000
    readable_time = datetime.fromtimestamp(timestamp_seconds).strftime('%Y-%m-%d %H:%M:%S.%f')
    # 获取当前时间戳（秒级）
    current_timestamp_seconds = time.time()
    current_readable_time = datetime.fromtimestamp(current_timestamp_seconds).strftime('%Y-%m-%d %H:%M:%S.%f')
    print(readable_time)
    print(current_readable_time)
    print(datas)
    if code in datas:
        tm = datas[code]['time']
        open = datas[code]['open']
        close = datas[code]['lastPrice']
        high = datas[code]['high']
        low = datas[code]['low']
        volume = datas[code]['volume']
        print(tm, open, high, low, close, volume)

        # 将数据添加到 DataFrame
        new_row = pd.DataFrame([{
            'code': code,
            'time': readable_time,
            'open': open,
            'close': close,
            'high': high,
            'low': low,
            'volume': volume
        }])

        df = pd.concat([df, new_row], ignore_index=True)
        print(df)


xtdata.subscribe_whole_quote(code_list=[code], callback=on_data)
#xtdata.subscribe_whole_quote(code_list=[code])
data = xtdata.get_market_data(['time', 'open', 'high', 'low', 'close', 'volume'], [code], period='1m', start_time='20240918')
# data = xtdata.get_market_data(['time', 'open', 'high', 'low', 'close', 'volume'], [code], period='1m')
print('result=====',data)
# {'000560.SZ': {'time': 1726210800000, 'lastPrice': 2.45, 'open': 2.4, 'high': 2.5100000000000002, 'low': 2.38, 'lastClose': 2.38, 'amount': 515111100.0, 'volume': 2100053, 'pvolume': 210005323, 'stockStatus': 0, 'openInt': 15, 'transactionNum': 0, 'lastSettlementPrice': 0.0, 'settlementPrice': 0.0, 'pe': 0.0, 'askPrice': [2.45, 2.46, 2.47, 0.0, 0.0], 'bidPrice': [2.44, 2.43, 2.42, 0.0, 0.0], 'askVol': [8792, 43077, 22965, 0, 0], 'bidVol': [21723, 20008, 11821, 0, 0], 'volRatio': 0.0, 'speed1Min': 0.0, 'speed5Min': 0.0}}
xtdata.run()