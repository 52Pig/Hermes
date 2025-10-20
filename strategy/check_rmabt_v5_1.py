# -*- coding: utf-8 -*-

import os
import asyncio
import glob
import json
import random
import datetime
import a_trade_calendar
import numpy as np
industry_dict = dict()


def get_latest_file(directory, prefix):
    # 获取目录下所有以 'prefix' 开头的文件
    files = [
        f for f in glob.glob(os.path.join(directory, f"{prefix}*"))
        if f.endswith('.json')  # 过滤条件
    ]
    # 如果没有找到文件，返回 None
    if not files:
        return None
    # print(files)
    # 从文件名中提取日期部分，并找到最新的文件
    def extract_date(file_path):
        # 示例文件名：reverse_moving_average_bull_track_20250303.json
        base_name = os.path.basename(file_path)  # 获取文件名部分
        date_str = base_name.split('_')[-1].split('.')[0]  # 分割出 YYYYMMDD
        return int(date_str)  # 转换为整数用于比较

    # 按日期降序排序后取最新文件
    files_sorted = sorted(files, key=extract_date, reverse=True)
    latest_file = files_sorted[0]
    latest_date = files_sorted[0].split("_")[-1].split(".")[0]
    print(f"[INIT]latest_file={latest_file}, latest_date={latest_date}")
    return latest_file, latest_date

def get_yesterday_date():
    """ 获取大A前一个交易日的日期 """
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    # print(today)
    previous_trade_date = a_trade_calendar.get_pre_trade_date(today).replace('-', '')
    return previous_trade_date

def load_stock_data(data_dir, target_codes, start_date=None, end_date=None):
    """加载所有目标股票的收盘价数据"""
    # 转换起始日期和结束日期为date对象
    print(start_date, end_date)

    if start_date is not None:
        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        elif isinstance(start_date, datetime.datetime):
            start_date = start_date.date()
    if end_date is not None:
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
        elif isinstance(end_date, datetime.datetime):
            end_date = end_date.date()

    files = []
    for f in os.listdir(data_dir):
        if f.startswith('hs_quant_base_') and f.endswith('.json'):
            date_str = f.split('_')[3].split('.')[0]
            date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
            files.append((date, os.path.join(data_dir, f)))
    files.sort(key=lambda x: x[0])

    # 根据日期范围过滤文件
    if start_date is not None or end_date is not None:
        files = [
            (date, path) for date, path in files
            if (start_date is None or date >= start_date) and
               (end_date is None or date <= end_date)
        ]
    print(f"files={files}")
    trading_dates = [date for date, _ in files]  # 提取回测区间内交易日

    stock_data = {code: {'dates': [], 'closes': [], 'volumes': [], 'highs': [], 'opens':[]} for code in target_codes}
    ind = 0
    for date, file_path in files:
        date_str = date.strftime('%Y-%m-%d')
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                ind += 1
                if ind % 50000 == 0:
                    print(f"processed data row_num = {ind}")
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except:
                    continue
                for code in data.keys():
                    if code in target_codes:
                        stock_info = data[code]
                        diff = 0
                        if 'd' in stock_info and stock_info['d']:
                            # 计算日级别成交量（所有分钟成交量之和）
                            day_volume = sum(minute_data[5] for minute_data in stock_info['d'][:-10])
                            # 获取当日14：50的收盘价
                            last_minute_data = stock_info['d'][-1]
                            if len(stock_info['d']) > 11:
                                last_minute_data = stock_info['d'][-10]
                            close_price = last_minute_data[2]
                            # if close_price > 40:    ## 只做低价
                            #     continue
                            open_price = stock_info['d'][0][1]
                            diff = close_price - open_price
                            # high_price = last_minute_data[3]
                            max_price = max(minute_data[3] for minute_data in stock_info['d'])

                            stock_data[code]['dates'].append(date_str)
                            stock_data[code]['closes'].append(close_price)
                            stock_data[code]['opens'].append(open_price)
                            stock_data[code]['highs'].append(max_price)
                            stock_data[code]['volumes'].append(day_volume)
                            # print(f'[INIT]load date={date_str}, code={code}, open={open_price}, close={close_price}, high={max_price}')

                        if 'i' in stock_info and stock_info['i']:

                            if date_str != str(start_date):
                                continue
                            print(date_str, start_date)
                            industry = stock_info['i'][2]
                            # print(industry, diff)
                            if industry not in industry_dict.keys():
                                industry_dict[industry] = [diff]
                            else:
                                industry_dict[industry].append(diff)
    return stock_data, trading_dates  # 返回股票数据和交易日列表





## 加载离线股票数据
pools_file, datestr = get_latest_file('D:/tool/dataset/strategy_data/', 'opt_rtrma_')
stock_dict = json.load(open(pools_file))

target_codes = list(stock_dict.keys())

## 加载下一个交易日的所有数据
stock_data, trading_dates = load_stock_data('D:/tool/dataset/quant_data/', target_codes, start_date='2025-10-16', end_date='2025-10-17')
# print(stock_data)
excel_path = r'D:/tool/dataset/temp/rtrma/analy_20251015.xlsx'


## 计算行业涨跌幅
# print(industry_dict)
statis_ind_dict = dict()
for ind, vvlist in industry_dict.items():
    degree = sum(vvlist) / len(vvlist)
    statis_ind_dict[ind] = degree

ind_dict = sorted(statis_ind_dict.items(), key=lambda x:x[1], reverse=True)
ind_res_dict = dict()
for ind, score in ind_dict:
    ind_res_dict[ind] = score
    # print(ind, score)


all_num = len(list(set(stock_dict.keys())))
recall = 0
pos = 0
neg = 0

records = []
ma5_k_dict = dict()
for kk, vv in stock_dict.items():
    # print(kk)
    idict = stock_dict.get(kk, {})
    # print(idict)
    code = idict.get("code", "")
    if code.startswith("688") or code.startswith("603"):
        continue
    is_target = idict.get("is_target", "")
    ## 受监控对象
    if "1" != is_target:
        continue
    # print(idict)
    off_close_str = idict.get("last_close")
    off_industry_str = idict.get("industry", "")
    off_last2d_close = float(off_close_str.split(',')[-2])
    off_close = float(off_close_str.split(',')[-1])

    if len(stock_data[kk]['dates']) <= 1:
        print(f"error code={code}")
        continue
    today_open = float(stock_data[kk]['opens'][0])
    today_close = float(stock_data[kk]['closes'][0])
    today_volume = float(stock_data[kk]['volumes'][0])
    # print("stock_data_kk====", stock_data[kk])
    next_close =  float(stock_data[kk]['closes'][1])
    off_close_list = [float(ele) for ele in off_close_str.split(',')]
    # print(off_close_list[-5:])
    last_1d_ma5 = np.mean(off_close_list[-5:])
    today_ma5 = np.mean([today_close] + off_close_list[-4:])
    print(code, last_1d_ma5, today_ma5)

    # if (off_close_list[-1] - off_close_list[-2]) / off_close_list[-2] > 0.04:
    #     continue
    # if (off_close_list[-2] - off_close_list[-3]) / off_close_list[-3] > 0.04:
    #     continue
    # if today_close/ - today_open >= 0 and today_close > off_last2d_open and today_volume < off_volume and today_volume > off_last2d_volume and -0.0056 <= (today_close - off_close) / off_close < 0.06:
    # if ( today_close - off_close ) / off_close > 0.06:
    #     continue
    # if ( today_close - off_close ) / off_close > 0:
    #     continue
    # if (today_ma5 / last_1d_ma5 - 1.0) < 0:
    #     continue
    ma5_k_dict[code] = (today_ma5 / last_1d_ma5 - 1.0)


ma5_k_sorted = sorted(ma5_k_dict.items(), key=lambda x:x[1], reverse=True)
tcodes = [cc for cc, _ in ma5_k_sorted[:1000]]
print(ma5_k_dict)
# quit()
for kk, vv in stock_dict.items():
    # print(kk)
    idict = stock_dict.get(kk, {})
    print(idict)
    code = idict.get("code", "")
    if code.startswith("688") or code.startswith("603")  or code not in tcodes:
        continue
    is_target = idict.get("is_target", "")
    ## 受监控对象
    if "1" != is_target:
        continue
    # print(idict)
    off_close_str = idict.get("last_close")
    off_volume_str = idict.get("last_volume")
    off_open_str = idict.get("last_open")
    off_industry_str = idict.get("industry", "")
    # if len(off_close_str) == 0 or len(off_volume_str) == 0 or len(off_industry_str) == 0:
    #     continue
    off_last2d_close = float(off_close_str.split(',')[-2])
    off_close = float(off_close_str.split(',')[-1])
    # off_last2d_open = float(off_open_str.split(',')[-2])
    off_volume = float(off_volume_str.split(",")[-1])
    off_last2d_volume = float(off_volume_str.split(",")[-2])
    off_last3d_volume = float(off_volume_str.split(",")[-3])

    #
    # for kstr, vstr in stock_data[kk].items():
    #     print(off_close, kstr, vstr)
        # date = stock_data[kk]['dates'][0]
        # close = stock_data[kk]['closes'][0]
        # volume = stock_data[kk]['volumes'][0]
        # high = stock_data[kk]['highs'][0]
        # open = stock_data[kk]['opens'][0]

    if len(stock_data[kk]['dates']) <= 1:
        print(f"error code={code}")
        continue
    today_open = float(stock_data[kk]['opens'][0])
    today_close = float(stock_data[kk]['closes'][0])
    today_volume = float(stock_data[kk]['volumes'][0])
    # print("stock_data_kk====", stock_data[kk])
    next_close =  float(stock_data[kk]['closes'][1])

    last_1d_ma5 = np.mean( [float(ele) for ele in off_close_str.split(',')][-5:]    )
    today_ma5 = np.mean([today_close] + [float(ele) for ele in off_close_str.split(',')][-4:])
    # if ind_res_dict.get(off_industry_str, 0) > 0.2 and today_close - today_open >= 0 and today_volume < off_volume and -0.0056 <= (today_close - off_close) / off_close < 0.06:
    # if today_close - today_open >= 0 and today_close > off_last2d_open and today_volume < off_volume and today_volume > off_last2d_volume and -0.0056 <= (today_close - off_close) / off_close < 0.06:

    # if today_close > off_close:
    # if today_close < today_open:
    print(code, today_close, today_open)
    ## 当天缩量幅度
    today_volume_zf = (today_volume - off_volume) / off_volume
    last_2d_volume_zf = (today_volume - off_last2d_volume) / off_last2d_volume
    last_3d_volume_zf = (today_volume - off_last3d_volume) / off_last3d_volume
    # if (today_close - today_open) / today_open < 0:
    if (today_close - today_open >= 0
            and today_volume_zf < -0.3 and last_2d_volume_zf < -0.3 and last_3d_volume_zf < -0.3
            and today_volume < off_volume < off_last2d_volume < off_last3d_volume
            and -0.0056 <= (today_close - off_close) / off_close < 0.06):
        recall += 1
        flag = 0
        if (next_close - today_close) > 0:
            pos+=1
            flag = 1
        else:
            neg+=1
        idict['flag'] = flag

        stock_dict[kk]['flag'] = 1 if next_close > today_close else 0
        stock_dict[kk]['last1d_close'] = off_close
        stock_dict[kk]['today_close'] = today_close
        stock_dict[kk]['next_close'] = next_close
        stock_dict[kk]['diff'] = next_close - today_close
        stock_dict[kk]['industry_score'] = ind_res_dict.get(off_industry_str, 0)
        # print(stock_dict[kk])
        records.append(stock_dict[kk])



# 3. 一次性写 Excel
import pandas as pd
df = pd.DataFrame(records)          # 如果提示列乱，可再 df = df[指定列顺序]

df.to_excel(excel_path, index=False)
print(f"已写入 Excel：{excel_path}")



res = dict()
res['all_num'] = all_num
res['recall_num'] = recall
res['pos'] = pos
res['neg'] = neg
print(json.dumps(res))