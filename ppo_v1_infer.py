import os, sys

# 设置项目根路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root_dir))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO
from ppo_v1_train import TradingEnvironment

import time
import random
import logging
import configparser
from datetime import datetime

from utils import utils
from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount
from xtquant import xtdata
from xtquant import xtconstant

if not os.path.exists("log"):
    os.makedirs('log')

# 设置日志
current_date = datetime.now().strftime('%Y-%m-%d')
logging.basicConfig(filename=f'log/ppo_v1_infer_{current_date}.log', level=logging.INFO, format='%(asctime)s %(message)s')


def infer_trading_strategy(model, data):
    env = TradingEnvironment(data)
    # env = DummyVecEnv([lambda: env])
    obs = env.reset()
    pred_list = []
    for _ in range(len(data)):
        action, a = model.predict(obs)
        print(action, a)
        obs, _, done, _ = env.step(action)
        pred_list.append(env.net_worth)
        if done:
            break
    return pred_list


def convert_to_floats(combined_str):
    try:
        # 分割字符串并转换为浮点数
        return [float(value) for value in combined_str.split(',')]
    except ValueError:
        # 如果转换失败，返回 NaN 或其他处理方式
        return [float('0.0')] * 506  # 根据列数返回 NaN 列表


class Hermes(object):
    def __init__(self):
        self.init_acc()
        self.init_model()

    def init_acc(self):
        # 初始化配置
        config = configparser.ConfigParser()
        config.read('conf/config.ini')
        # config.read(['conf/config.ini'])
        path = config.get("client", 'mini_path')
        acc_name=***
        print('[mini_path]', path, acc_name)
        session_id = int(random.randint(100000, 999999))
        xt_trader = XtQuantTrader(path, session_id)
        # 链接qmt客户端
        xt_trader.start()
        connect_result = xt_trader.connect()
        # 订阅账户
        acc = StockAccount(acc_name)
        subscribe_res = xt_trader.subscribe(acc)
        self.xt_trader = xt_trader
        self.acc = acc
        print('[DEBUG]connect_status=', connect_result, ',subscribe_status=', subscribe_res)

    def init_model(self):
        model_dir = "model_dir/model_81000.zip"
        model = PPO.load(model_dir)

        fname = './dataset/000560.SZ'
        data = pd.read_csv(fname, sep="\t")
        # column_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        data = data.rename(columns={data.columns[0]: 'Date'})
        data = data.rename(columns={data.columns[1]: 'Open'})
        data = data.rename(columns={data.columns[2]: 'High'})
        data = data.rename(columns={data.columns[3]: 'Low'})
        data = data.rename(columns={data.columns[4]: 'Close'})
        data = data.rename(columns={data.columns[5]: 'Volume'})
        data = data.rename(columns={data.columns[6]: 'StockHash'})

        data = data.rename(columns={data.columns[7]: 'year_day'})
        data = data.rename(columns={data.columns[8]: 'year_month'})
        data = data.rename(columns={data.columns[9]: 'month_day'})
        data = data.rename(columns={data.columns[10]: 'week_day'})
        data = data.rename(columns={data.columns[11]: 'hour_of_day'})
        data = data.rename(columns={data.columns[12]: 'minute_of_hour'})

        # 删除第7列（索引6）
        data = data.drop(data.columns[6], axis=1)
        print(data.columns)

        # 合并列，并使用逗号分隔
        data['combined'] = data[
            ['year_day', 'year_month', 'month_day', 'week_day', 'hour_of_day', 'minute_of_hour']].astype(str).agg(
            ','.join,
            axis=1)

        # 定义转换函数
        def convert_to_floats(combined_str):
            try:
                # 分割字符串并转换为浮点数
                return [float(value) for value in combined_str.split(',')]
            except ValueError:
                # 如果转换失败，返回 NaN 或其他处理方式
                return [float('0.0')] * 506  # 根据列数返回 NaN 列表

        # 应用转换函数
        floats_data = data['combined'].apply(convert_to_floats)

        # 调试信息：打印前几行
        # print(floats_data.head())

        # 检查列表长度
        list_lengths = floats_data.apply(len)
        print("List lengths:", list_lengths.unique())

        # 确保所有列表长度相同，并创建 DataFrame
        if list_lengths.nunique() == 1:
            floats_df = pd.DataFrame(floats_data.tolist())  # ,
            # columns=['year_day', 'year_month', 'month_day', 'week_day', 'hour_of_day',
            #         'minute_of_hour'])
            data = pd.concat([data, floats_df], axis=1)
        else:
            print("Error: Not all lists have the same length.")

        # 删除原始的合并列
        data = data.drop(columns=['combined'])

        # 转换日期列为时间戳
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data['Date'] = data['Date'].astype(np.int64) / 10 ** 9  # 转换为时间戳（秒）

        # 确保列名是字符串
        data.columns = data.columns.astype(str)
        # 确保所有数据列都是数值型
        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.fillna(0)  # 或者使用其他合适的填充值

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0.01, 1))
        data_normed = scaler.fit_transform(data)
        # convert data into a pandas DataFrame
        data = pd.DataFrame(data_normed, columns=data.columns)
        train_size = int(len(data))
        train_data = data[:train_size]
        # Create the trading environment

        env = TradingEnvironment(train_data)
        self.env = env
        # env = DummyVecEnv([lambda: env])
        obs = env.reset()
        self.model = model
        self.obs = obs
        self.env = env

    def feature_process(self, data):
        print()

    def do(self):
        print()


def get_current_time():
    # return datetime.now().strftime('%Y%m%d %H:%M:%S')
    return datetime.now().strftime('%Y%m%d%H%M%S')


if __name__ == "__main__":
    #os.chdir('D:/tool/pycharm_client/workspace/Hermes')
    sys = Hermes()
    ## 订阅当天所有实时行情
    period = 'tick'
    stock_code = '000560.SZ'
    # stock_code = '603883.SH'
    # stock_code = '000948.SZ'
    # xtdata.subscribe_quote(stock_code, period=period, count=-1)  # 设置count = -1来取到当天所有实时行情
    # 下载每支股票数据
    field_list = ['time', 'open', 'high', 'low', 'close', 'volume']
#    is_down_suc = xtdata.download_history_data2(stock_list=[stock_code], period='1m', start_time='20240910093000', end_time='20240920093000')
#     is_down_suc = xtdata.download_history_data2(stock_list=[stock_code], period='1m', start_time='20240924093000')
#     xtdata.download_history_data(stock_code, '1m', '20240922')
    # quit()
    while True:
        xtdata.download_history_data(stock_code, '1m', '20240101')
        current_time = datetime.now().time()
        start_time = datetime.strptime("09:31", "%H:%M").time()
        mid_start_time = datetime.strptime("11:30", "%H:%M").time()
        mid_end_time = datetime.strptime("13:01", "%H:%M").time()
        end_time = datetime.strptime("14:55", "%H:%M").time()

        # 检查当前时间是否在 9:30 到 15:00 之间
        if start_time <= current_time <= mid_start_time or mid_end_time <= current_time <= end_time:
            field_list = ['time', 'open', 'high', 'low', 'close', 'volume']
            # kline_data = xtdata.get_market_data_ex(field_list=field_list, stock_list=[stock_code], period='1m')
            # kline_data = xtdata.get_market_data_ex(field_list=field_list, stock_list=[stock_code], period='tick')
            kline_data = xtdata.get_market_data_ex(field_list=field_list, stock_list=[stock_code], period='1m', start_time='20240923093000')


            # print(kline_data)
            df = kline_data[stock_code]
            # 获取最大时间戳的行
            max_row = df.loc[df['time'].idxmax()]
            open_price = float(max_row["open"])
            high_price = float(max_row["high"])
            low_price = float(max_row["low"])
            close_price = float(max_row["close"])
            volume = float(max_row["volume"])

            # 将时间戳转换为日期时间格式
            timestamp = max_row['time']
            dtime = pd.to_datetime(timestamp, unit="ms").tz_localize('UTC').tz_convert('Asia/Shanghai')

            # print(datetime)
            dt, year_day, year_month, month_day, week_day, hour_of_day, minute_of_hour = utils.parse_time(str(dtime))
            print(dt, year_day, year_month, month_day, week_day, hour_of_day, minute_of_hour)

            year_day_oh = utils.to_onehot(year_day, 367)
            year_day_list = [ele for ele in year_day_oh]
            year_month_oh = utils.to_onehot(year_month, 13)
            year_month_list = [ele for ele in year_month_oh]
            month_day_oh = utils.to_onehot(month_day, 32)
            month_day_list = [ele for ele in month_day_oh]
            week_day_oh = utils.to_onehot(week_day, 8)
            week_day_list = [ele for ele in week_day_oh]
            hour_of_day_oh = utils.to_onehot(hour_of_day, 25)
            hour_of_day_list = [ele for ele in hour_of_day_oh]
            minute_of_hour_oh = utils.to_onehot(minute_of_hour, 61)
            minute_of_hour_list = [ele for ele in minute_of_hour_oh]
            tm_seq = year_day_list + year_month_list + month_day_list + week_day_list + hour_of_day_list + minute_of_hour_list

            row_line = [dt, open_price, high_price, low_price, close_price, volume] + tm_seq
            feature_size = len(row_line)
            # 创建一个空的 DataFrame
            #columns = ['date', 'open', 'high', 'low', 'close', 'volume']
           # 创建新的 DataFrame，列数为 row_data 的长度
            feature = pd.DataFrame(columns=range(feature_size))
            # 添加行数据
            feature.loc[len(feature)] = row_line
            # print(feature)
            # quit()

            feature = feature.rename(columns={feature.columns[0]: 'Date'})
            feature = feature.rename(columns={feature.columns[1]: 'Open'})
            feature = feature.rename(columns={feature.columns[2]: 'High'})
            feature = feature.rename(columns={feature.columns[3]: 'Low'})
            feature = feature.rename(columns={feature.columns[4]: 'Close'})
            feature = feature.rename(columns={feature.columns[5]: 'Volume'})
            # feature = feature.rename(columns={feature.columns[6]: 'StockHash'})

            feature = feature.rename(columns={feature.columns[6]: 'year_day'})
            feature = feature.rename(columns={feature.columns[7]: 'year_month'})
            feature = feature.rename(columns={feature.columns[8]: 'month_day'})
            feature = feature.rename(columns={feature.columns[9]: 'week_day'})
            feature = feature.rename(columns={feature.columns[10]: 'hour_of_day'})
            feature = feature.rename(columns={feature.columns[11]: 'minute_of_hour'})

            # 转换日期列为时间戳
            if 'Date' in feature.columns:
                feature['Date'] = pd.to_datetime(feature['Date'])
                feature['Date'] = feature['Date'].astype(np.int64) / 10 ** 9  # 转换为时间戳（秒）

            # # 假设 feature 已经存在
            # num_new_columns = 6
            # # 添加6列，列名为 'new_col_0', 'new_col_1', ..., 'new_col_5'
            # for i in range(num_new_columns):
            #     feature[f'extra_{i}'] = 0

            # print(feature)
            # 确保列名是字符串
            feature.columns = feature.columns.astype(str)
            # 确保所有数据列都是数值型
            feature = feature.apply(pd.to_numeric, errors='coerce')
            feature = feature.fillna(0)  # 或者使用其他合适的填充值

            # print(len(feature), feature)
            # Normalize the data
            scaler = MinMaxScaler(feature_range=(0.01, 1))
            data_normed = scaler.fit_transform(feature)
            # convert data into a pandas DataFrame
            infer_data = pd.DataFrame(data_normed, columns=feature.columns)

            # 将新数据添加到环境中
            sys.env.data = pd.concat([sys.env.data, infer_data], ignore_index=True)
            action, _ = sys.model.predict(sys.obs)
            # 0:buy,1:sell,2:hold
            # print(action)
            current_price = 0
            order_id = -1


            if 0 == action:
                print("buy")
                # 查询账户余额
                acc_info = sys.xt_trader.query_stock_asset(sys.acc)
                cash = acc_info.cash
                # 查询当前股价
                current_price = utils.get_latest_price(stock_code)

                ## 若账户余额> 股票价格*100，则买入
                # 下单
                if current_price is not None:
                    if cash < current_price * 100:
                        continue
                    order_id = sys.xt_trader.order_stock(sys.acc, stock_code, xtconstant.STOCK_BUY, 100, xtconstant.FIX_PRICE, current_price)
                    #print('[DEBUG]buy=', current_price, order_id)

            elif 1 == action:
                print("sell")
                # 查询持仓市值
                acc_info = sys.xt_trader.query_stock_asset(sys.acc)
                marketValue = acc_info.m_dMarketValue
                # 查询当前股价
                current_price = utils.get_latest_price(stock_code)
                # 卖出
                if current_price is not None:
                    # if marketValue < current_price * 100:
                    if marketValue <= 0:
                        continue
                    order_id = sys.xt_trader.order_stock(sys.acc, stock_code, xtconstant.STOCK_SELL, 100, xtconstant.FIX_PRICE, current_price)
                    #print(order_id)
            else:
                print("hold")
                # self.xt_trader.cancel_order_stock(self.acc, order_id)

            # sys.obs, reward, done, _ = sys.env.step(action)
            obs, reward, done, _ = sys.env.step(action)
            # 在推理之前检查观察形状
            if obs.shape[0] > 518:
                sys.obs = obs[:518]  # 截取前 518 个元素

            if order_id != -1:
                acc_info = sys.xt_trader.query_stock_asset(sys.acc)
                total_asset = acc_info.total_asset
                print('[DEBUG]result=', current_price, action, order_id, reward, sys.env.net_worth, total_asset)
                logging.info(f"{stock_code},{current_price},{action},{order_id},{reward},{sys.env.net_worth},{total_asset}")
            # # Plot the net worth over time
            # plt.plot(worth_list)
            # plt.xlabel("Time")
            # plt.ylabel("Net Worth")
            # plt.title("Net Worth over Time")
            # plt.show()

        ## 10秒检测一次
        time.sleep(65)


    # # 合并列，并使用逗号分隔
    # data['combined'] = data[
    #     ['year_day', 'year_month', 'month_day', 'week_day', 'hour_of_day', 'minute_of_hour']].astype(str).agg(','.join,
    #                                                                                                           axis=1)
    # # 应用转换函数
    # floats_data = data['combined'].apply(convert_to_floats)
    #
    # # 调试信息：打印前几行
    # print(floats_data.head())

    # # 检查列表长度
    # list_lengths = floats_data.apply(len)
    # print("List lengths:", list_lengths.unique())

    # # 确保所有列表长度相同，并创建 DataFrame
    # if list_lengths.nunique() == 1:
    #     floats_df = pd.DataFrame(floats_data.tolist())#,
    #                              #columns=['year_day', 'year_month', 'month_day', 'week_day', 'hour_of_day',
    #                              #         'minute_of_hour'])
    #     data = pd.concat([data, floats_df], axis=1)
    # else:
    #     print("Error: Not all lists have the same length.")

    # # 删除原始的合并列
    # data = data.drop(columns=['combined'])
    #


