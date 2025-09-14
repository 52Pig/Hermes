# -*- coding: utf-8 -*-

import os
import asyncio
import glob
import json
import random
import datetime
import logging
import a_trade_calendar
# from sympy import false
from xtquant import xtconstant
from xtquant import xtdata
import threading
from queue import Queue

from base_strategy import BaseStrategy
from utils import utils
current_date = datetime.datetime.now().strftime('%Y-%m-%d')
pp = "E:/tool/pycharm_client/workspace/Hermes/logs"  # 修正这个路径的拼写错误

# 检查并创建日志目录
if not os.path.exists(pp):
    os.makedirs(pp, exist_ok=True)

log_file = os.path.join(pp, f"rmabt_v1_{current_date}.log")
print(log_file)

from queue import Queue as LogQueue

# # 全局日志队列
# log_queue = LogQueue()
# queue_handler = logging.handlers.QueueHandler(log_queue)
# queue_listener = logging.handlers.QueueListener(
#     log_queue,
#     logging.FileHandler(log_file, mode='a', encoding='utf-8'),
#     logging.StreamHandler()
# )
# queue_listener.start()
#
# # 配置日志记录器
# rma_logger = logging.getLogger("ReverseMABULL")
# rma_logger.setLevel(logging.INFO)
# rma_logger.addHandler(queue_handler)  # 替代原有FileHandler


# 创建一个新的日志记录器
rma_logger = logging.getLogger("ReverseMABULL")
rma_logger.setLevel(logging.INFO)

# 检查是否已存在FileHandler，避免重复添加
file_handler_exists = any(
    isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(log_file)
    for handler in rma_logger.handlers
)

if not file_handler_exists:
    # 创建文件处理器，使用追加模式('a')
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    file_handler.setFormatter(formatter)
    rma_logger.addHandler(file_handler)

# 示例日志记录
rma_logger.info("服务启动，日志已配置为追加模式。")

class ReverseMABULLV2(BaseStrategy):
    def update_config(self, new_config):
        """动态更新配置参数"""
        self.config = new_config
        # 可添加特定参数的更新逻辑（如调整阈值）

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # self.data_queue = Queue()  # 用于接收行情数据的队列
        # self.is_running = True  # 控制线程运行状态

        self.is_test = "0"
        ## 记录买入委托，已经委托订单不再进行重复下单
        self.wt_dict = dict()
        self.sell_price_history = {}
        self.buy_price_history = {}
        # self.code_max_price = {}
        self.sorted_gpzt_pools = list()
        # pools_file = self.get_latest_file('./logs', 'reverse_moving_average_bull_track_')
        pools_file = self.get_latest_file('F:/tools/dataset/strategy_data', 'reverse_moving_average_bull_track_')
        # pools_file = './logs/reverse_moving_average_bull_track_20250419.json'
        self.stock_dict = json.load(open(pools_file))
        self.pools_list = list()
        keys_list = list(self.stock_dict.keys())
        # random.shuffle(keys_list)
        # for kk in keys_list:
        for kk, vv in self.stock_dict.items():
            # print(kk)
            idict = self.stock_dict.get(kk, {})
            cc = idict.get("code", "")
            # last_close_str = idict.get("last_close", "")
            is_target = idict.get("is_target", "")
            if kk == "603928":
                print(is_target, idict)
            # if len(cc) > 0 and len(self.pools_list)<60 and is_target == "1":
            if len(cc) > 0 and (is_target == "1" or is_target == 1):
                self.pools_list.append(cc)
        if "603928" in self.pools_list:
            print('++++++++++++++++++++++')
        print('[INIT]load pools file name:', pools_file)
        print('[INIT]load total size:', len(keys_list), keys_list)
        print('[INIT]load target size:', len(self.pools_list), self.pools_list[:5])
        print('[INIT]SUCCEED!')

    def get_latest_file(self, directory, prefix):
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
        # latest_file = files_sorted[1]

        # latest_file = max(files, key=lambda x: x.split('.')[-1])  # 按日期部分比较文件名
        print(f"[INIT]latest_file={latest_file}")
        return latest_file

    def get_buy_volume(self, price):
        """
            根据股票价格和目标总金额，计算应购买的股票数量（按整手计算，1手=100股）
            :param price: 股票价格（1.0~30.0元）
            :param target_amount: 目标总金额（如10000元）
            :return: 股票数量（整百股数）
        """
        target_amount = 10000
        if price <= 0 or target_amount <= 0:
            return 0  # 处理非法输入
        # 计算理想手数（可能含小数）
        ideal_hands = target_amount / (price * 100)
        # 获取候选手数（地板值和天花板值）
        floor_hands = int(ideal_hands)
        ceil_hands = floor_hands + 1

        # 计算两种手数的实际总金额
        amount_floor = floor_hands * price * 100
        amount_ceil = ceil_hands * price * 100

        # 比较哪个更接近目标金额（优先选择不超支的方案）
        diff_floor = abs(target_amount - amount_floor)
        diff_ceil = abs(target_amount - amount_ceil)

        # 如果差距相等，优先选择金额较小的方案（如9000 vs 11000时选9000）
        if diff_floor <= diff_ceil:
            return floor_hands * 100
        else:
            return ceil_hands * 100

    def should_sell(self, stock, last_1d_close_price, max_price, max_price_timestamp, last_price, bid_price, bid_vol, has_open_price):
        """判断是否满足卖出条件"""
        ## 卖出情况0：止盈位,一天内利润8%
        # if (last_price - last_1d_close_price) / last_1d_close_price > 0.08:
        #     return True, 0

        ## 卖：收益3%以上，冲高回落60%以上则离场
        if (last_price - has_open_price) / has_open_price > 0.033 and last_price < max_price * 0.94:
            return True, -2


        ## 卖出情况0_2: 止盈位 ，封板但是封单量<2w
        zt_price = round(last_1d_close_price * 1.1, 2)
        if zt_price == bid_price[0] and bid_vol[0] < 30000:
            return True, -1
        ## 卖出情况1: 止损位，一天内亏损6.2%
        if (last_price - last_1d_close_price) / last_1d_close_price < -0.062:
            return True, 1

        ## 止损，多天总共亏损成本价的-3%
        # if has_open_price > 0 and (last_price - has_open_price) / has_open_price < -0.058:
        #     return True, 7

        ## 卖出情况2: ma5/10平行或向下
        info_dict = self.stock_dict.get(stock.split('.')[0], {})
        last_close_str = info_dict.get("last_close", "")
        last_close_list = last_close_str.split(",")
        if "" != last_close_str and len(last_close_list) > 32:
            # 利用最新价格计算当天的MA5，MA10，MA20，MA30
            lc_list = [float(ele) for ele in last_close_list]
            lc_list.append(last_price)
            len_lc_list = len(lc_list)
            ma5 = sum(lc_list[len_lc_list - 5:]) / 5.0
            prev_ma5 = info_dict.get("MA5", 1000)
            # ma5无上升趋势
            increase_ma5 = (ma5 <= prev_ma5)
            ma5_rate = ( ma5 - prev_ma5 ) / prev_ma5
            if increase_ma5:
                return True, 6
            # if increase_ma5 and ma5_rate < -0.007:
            #     return True, 6


            ## 高位离场
            last2d_ma5 = sum(lc_list[:-1][len(lc_list[:-1]) - 5:]) / 5.0
            curr_k = (ma5 - prev_ma5) / prev_ma5
            last1d_k = (prev_ma5 - last2d_ma5) / last2d_ma5
            ma5_k = (curr_k - last1d_k) / (last1d_k + 0.00001)
            ## 止盈
            if has_open_price > 0 and last_price > has_open_price * 1.2 and ma5_k < -0.2:
                return True, 8



            # ma10 = sum(lc_list[len_lc_list - 10:]) / 10.0
            # prev_ma10 = info_dict.get("MA10", 1000)
            # # ma10无上升趋势
            # increase_ma10 = (ma10 <= prev_ma10)
            # if increase_ma10:
            #     return True, 7

        # 卖出情况3：若当天max_price>0.095后超过5分钟股价不再>0.095则卖出
        if max_price_timestamp is not None and max_price is not None:
            limit_up_price = round(last_1d_close_price * 1.095, 2)
            current_time = datetime.datetime.now()
            time_diff = (current_time - max_price_timestamp).total_seconds()
            if time_diff > 300 and max_price >= limit_up_price > last_price:
                return True, 2
        # 卖出情况4：检查最近3次股价是否全部低于前一天收盘价的6.2%
        if stock not in self.sell_price_history or len(self.sell_price_history[stock]) < 3:
            return False, 3  # 数据不足

        his_price_list = self.sell_price_history[stock]
        lt_target_num = 0
        if len(his_price_list) > 3:
            his_price_list = his_price_list[:3]
        for price in his_price_list:
            if ( price - last_1d_close_price ) / last_1d_close_price < -0.062:
                lt_target_num += 1
        if lt_target_num == 3:
            return True, 4
        else:
            return False, 5

    def should_buy(self, stock_code, current_price, last_1d_close_price, info_dict, is_high_fx, is_noon_time, max_price):
        ## 当天涨幅>x%不再追高，即使买入溢价不高，反而可能诱多跌幅较大
        if (current_price - last_1d_close_price ) / last_1d_close_price > 0.067:
            return False, 199

        if max_price > 0 and max_price == current_price:
            return False, 198

        ## 记录了最近5次的历史价格数据
        his_price_list = self.buy_price_history[stock_code]
        if len(his_price_list) > 5:
            his_price_list = his_price_list[:5]
        # # 连续满足条件的判断，确保股票价格有上涨的连续性，避免在趋势不明时买入
        # if is_high_fx and len([price for price in his_price_list if price < current_price]) < 3:
        #     return False, 200

        # 多头排列&上升趋势
        last_close_str = info_dict.get("last_close", "")
        last_close_list = last_close_str.split(",")
        if "" == last_close_str or len(last_close_list) < 32:
            return False, 201
        else:
            # 利用最新价格计算当天的MA5，MA10，MA20，MA30
            lc_list = [float(ele) for ele in last_close_list]
            lc_list.append(current_price)
            len_lc_list = len(lc_list)
            ma5 = sum(lc_list[len_lc_list - 5:]) / 5.0
            ma10 = sum(lc_list[len_lc_list - 10:]) / 10.0
            ma20 = sum(lc_list[len_lc_list - 20:]) / 20.0
            ma30 = sum(lc_list[len_lc_list - 30:]) / 30.0
            prev_ma5 = info_dict.get("MA5", 1000)
            prev_ma10 = info_dict.get("MA10", 1000)
            prev_ma20 = info_dict.get("MA20", 1000)
            prev_ma30 = info_dict.get("MA30", 1000)
            # 上升趋势
            # increase_status = (ma5 > prev_ma5) and (ma10 >= prev_ma10) and (ma20 >= prev_ma20) and (ma30 >= prev_ma30)
            # increase_status = (ma5 > prev_ma5) and (ma10 > prev_ma10)    ## 这样大部分是亏的
            ma5_rate = (ma5 - prev_ma5) / prev_ma5
            ma10_rate = (ma10 - prev_ma10) / prev_ma10
            ma20_rate = (ma20 - prev_ma20) / prev_ma20
            ma30_rate = (ma30 - prev_ma30) / prev_ma30

            # increase_status = (ma5 > prev_ma5) and ma10_rate > -0.01 and ma20_rate > -0.006 and ma30_rate > -0.009
            # increase_status = ma5_rate >= 0.005 and ma10_rate > -0.01 and ma20_rate > -0.006 and ma30_rate > -0.009

            increase_status = ma5_rate >= 0.005 and ma10_rate > 0.0025 and ma20_rate > 0.001 and ma30_rate >= 0

            # ma5非最低
            is_ma5_not_lowest = ma5 >= ma10 or ma5 >= ma20 or ma5 >= ma30
            # is_ma5_not_lowest = ma5 > ma10 or ma5 > ma20 or ma5 > ma30
            if increase_status and is_ma5_not_lowest:
                return True, 202
            else:
                return False, 203

    async def do(self, accounts):
        print("[DEBUG]do reverse_moving_average_bull_track v2", utils.get_current_time(), accounts)
        target_code = '沪深A股'
        # req_dict = accounts.get("acc_1", {})
        req_dict = accounts.get("acc_2", {})
        xt_trader = req_dict.get("xt_trader")
        acc = req_dict.get("account")
        is_test = req_dict.get("is_test", "0")
        self.is_test = is_test
        # acc_name = req_dict.get("acc_name")

        ## 加载有效召回池
        if len(self.pools_list) == 0:
            return json.dumps({"warn":[{"mark":"pools_list is empty."}]})

        ## 辅助时间
        cur_time = datetime.datetime.now().time()
        gy_time = datetime.datetime.strptime("22:01", "%H:%M").time()
        jj_start_time = datetime.datetime.strptime("09:15", "%H:%M").time()
        jj_end_time = datetime.datetime.strptime("09:19", "%H:%M").time()
        start_time = datetime.datetime.strptime("09:32", "%H:%M").time()
        mid_start_time = datetime.datetime.strptime("11:30", "%H:%M").time()
        mid_end_time = datetime.datetime.strptime("13:01", "%H:%M").time()
        end_time = datetime.datetime.strptime("14:55", "%H:%M").time()
        high_fx_time = datetime.datetime.strptime("10:25", "%H:%M").time()
        after_noon_time = datetime.datetime.strptime("14:00", "%H:%M").time()
        is_high_fx = cur_time <= high_fx_time
        is_noon_time = cur_time > after_noon_time
        is_trade_time = start_time <= cur_time <= mid_start_time or mid_end_time <= cur_time <= end_time
        is_jj_time = jj_start_time <= cur_time <= jj_end_time

        ## 不在交易时间不操作
        if not is_trade_time and self.is_test == "0":
           return json.dumps({"warn":[{"mark":"is_not_trade_time."}]})


        ## 持仓行业分布
        industry_num_map = dict()
        # 已经持仓股票也需要放到订阅中
        has_stock_obj = xt_trader.query_stock_positions(acc)
        has_stock_list = list()
        has_stock_map = dict()
        stock_list = self.pools_list
        for has_stock in has_stock_obj:
            # print('持仓总市值=market_value=', has_stock.market_value)
            # print('成本=open_price=', has_stock.open_price)
            has_volume = has_stock.volume
            if 0 == has_volume:
                continue
            has_stock_code = has_stock.stock_code
            has_open_price = has_stock.open_price
            has_stock_map[has_stock_code] = {}
            has_stock_map[has_stock_code]['volume'] = has_volume
            has_stock_map[has_stock_code]['open_price'] = has_open_price
            has_stock_list.append(has_stock_code)

            industry = self.stock_dict.get(has_stock_code.split('.')[0], {}).get("industry", "")
            if '' != industry:
                if industry in industry_num_map:
                    industry_num_map[industry] += 1
                else:
                    industry_num_map[industry] = 1
        # subscribe_whole_list = [ele.split('.')[0] for ele in set(has_stock_list + stock_list) if ele.split('.')[0].isdigit()]
        # print(f"[DEBUG]rma subscribe_whole_list={len(subscribe_whole_list)}={subscribe_whole_list}")
        ## 查询实时价格
        # df = utils.get_stock_now_data(code_list=subscribe_whole_list)
        # if df is None:
        #     return json.dumps({"warn": [{"mark": "realtime data frame is None."}]})
        # df = df.dropna(subset=['最新价', '今开'])
        # print(df)

        subscribe_whole_list = list(set(has_stock_list + stock_list))
        df = xtdata.get_full_tick(subscribe_whole_list)
        # rma_logger.info(f"Tick Data Structure: {json.dumps(df, ensure_ascii=False)}")
        # print(json.dumps(df, indent=2))
        #### 0.已经持仓的股票是否要卖出
        for code in has_stock_list:
            stock = code
            has_volume = has_stock_map[stock].get("volume", 0)
            has_open_price = has_stock_map[stock].get("open_price", 0)
            code = code.split('.')[0]
            # print("___________sell_____________", code)
            info_dict = self.stock_dict.get(code, {})
            last_close_list = [float(ele) for ele in info_dict.get("last_close", "").split(",") if ele]
            latest_price = 0.0
            open_price = 0.0
            max_price = 0.0
            bid_price = [0, 0, 0, 0, 0]
            bid_vol = [0, 0, 0, 0, 0]
            # 从实时数据获取最新价格
            try:
                # 注意：需要确认代码格式是否匹配（如是否包含交易所后缀）
                # stock_row = df[df['代码'] == code].iloc[0]  # 假设代码精确匹配
                # open_price = float(stock_row['今开'])
                # max_price = float(stock_row['最高'])
                # max_price = float(stock_row['今开'])
                # latest_price = float(stock_row['最新价'])
                # print(code, latest_price, stock_row)

                stock_row = df.get(stock)
                open_price = float(stock_row['open'])
                max_price = float(stock_row['high'])
                latest_price = float(stock_row['lastPrice'])
                bid_price = stock_row['bidPrice']
                bid_vol = stock_row['bidVol']
                # print('sell', stock, latest_price, bid_price, bid_vol)
            except IndexError:
                # print(f"代码 {code} 未在实时数据中找到，跳过处理")
                continue
            except KeyError:
                # print(f"实时数据中缺少必要字段，请检查数据源")
                continue
            except Exception as e:
                # print(f"异常：{e}")
                continue
            if latest_price is None or latest_price == 0:
                continue

            # 保留最近5次的股价数据
            if stock not in self.sell_price_history:
                self.sell_price_history[stock] = list()
            self.sell_price_history[stock].append(latest_price)
            if len(self.sell_price_history[stock]) > 5:
                self.sell_price_history[stock].pop(0)

            max_price_timestamp = None
            # 记录最高价时间戳
            if latest_price <= max_price and max_price_timestamp is None:
                max_price_timestamp = datetime.datetime.now()
            last_1d_close_price = last_close_list[-1]

            is_sell, sell_id = self.should_sell(stock, last_1d_close_price, max_price, max_price_timestamp, latest_price, bid_price, bid_vol, has_open_price)
            if is_sell and has_volume > 0:
                # 为了避免无法出逃，价格笼子限制，卖出价格不能低于当前价格的98%
                sell_price = round(latest_price * 0.99, 2)
                if sell_price < round(last_1d_close_price - last_1d_close_price * 0.1, 2):
                    sell_price = round(last_1d_close_price - last_1d_close_price * 0.1, 2)
                order_id = -1
                if "1" != self.is_test:
                    order_id = xt_trader.order_stock(acc, stock, xtconstant.STOCK_SELL, has_volume,
                                                xtconstant.FIX_PRICE, sell_price)
                else:
                    print(f"[DEBUG]is_test=1,sell_stock={stock},sell_price={sell_price}，sell_volume={has_volume}")

                sell = dict()
                sell['code'] = stock
                sell['price'] = sell_price
                sell['action'] = 'sell'
                sell['order_id'] = order_id
                sell['volume'] = has_volume
                sell["sell_id"] = sell_id
                rma_logger.info("s:" + json.dumps(sell, ensure_ascii=False))
                if "1" == self.is_test:
                    print("s:" + json.dumps(sell, ensure_ascii=False))



        ## 动态判断是否要开仓
        # min_hold = 5    # 最小持仓
        # max_hold = 12   # 最大持仓
        min_hold = 5
        max_hold = 24
        current_num = len(has_stock_list)
        target_num = 0    # 所需补充量
        if current_num < min_hold:
            target_num = min_hold - current_num
        else:
            # 检查所有持仓是否均为正收益
            all_positive = 0
            for stock, has_info in has_stock_map.items():
                try:
                    has_open_price = has_info.get("open_price", -1)
                    stock_row = df.get(stock)
                    latest_price = float(stock_row['lastPrice'])
                    if has_open_price < latest_price:
                        all_positive += 1
                except (KeyError, ValueError, TypeError):
                    pass
            if all_positive == current_num and current_num < max_hold:
                target_num = 1
        # if len(has_stock_list) > 10:
        # print("1111111111111", target_num, current_num)
        target_num = min(target_num, max_hold - current_num)
        # print("22222222222", target_num, current_num)
        target_num = max(target_num, 0)
        # print("333333333333333", target_num)
        # if target_num == 0:
        #     return json.dumps({"warn": [{"mark": "suppliment target 0"}]})

        #### 1.计算每支打分
        # 查询账户委托
        ## 当前是否有委托单,避免重复报废单
        stock_wt_map = dict()
        wt_infos = xt_trader.query_stock_orders(acc, True)
        for wt_info in wt_infos:
            print(f'委托信息：wt_code={wt_info.stock_code},wt_volume={wt_info.order_volume}, wt_price={wt_info.price}, wt_info={wt_info}s')
            if wt_info.stock_code is not None:
                stock_wt_map[wt_info.stock_code] = 1
                ## 加入行业分布中
                code = wt_info.stock_code.split('.')[0]
                industry = self.stock_dict.get(code, {}).get("industry","")
                if industry in industry_num_map:
                    industry_num_map[industry] += 1
                else:
                    industry_num_map[industry] = 1

        ## 查询当前账户余额
        acc_info = xt_trader.query_stock_asset(acc)
        cash = 0
        if acc_info is not None:
            cash = acc_info.cash


        code_score_dict = dict()
        for code in self.pools_list:
            ## 已经持仓不再买入，pools_list中的code带有后缀，stock_wt_map中的code带有后缀，has_stock_list带有后缀
            # print(f"[DEBUG]rma before buy, pools code={code};wt_code_list={stock_wt_map};has_stock_list={has_stock_list}")
            if code in has_stock_list:
                continue
            ## 已经委托不再买入
            if stock_wt_map.get(code, 0) > 0 or self.wt_dict.get(code, 0) > 0:
                continue
            ## 已经持仓了行业数量>3，则不再买入
            cc = code.split('.')[0]
            info_dict = self.stock_dict.get(cc, {})
            industry = info_dict.get("industry", "")
            stock = info_dict.get("code", "")
            # print(f"[DEBUG]rma code={code},industry={industry},industry_map={json.dumps(industry_num_map, ensure_ascii=False)}")
            ## 暂时不做行业限制，回测时也没限制
            # if "" != industry:
            #     if industry_num_map.get(industry, 0) >= 3:
            #         continue

            last_close_list = [float(ele) for ele in info_dict.get("last_close", "").split(",") if ele]
            latest_price = 0.0
            # 从实时数据获取最新价格
            try:
                # 注意：需要确认代码格式是否匹配（如是否包含交易所后缀）
                # stock_row = df[df['代码'] == code].iloc[0]  # 假设代码精确匹配
                # latest_price = stock_row['最新价']

                stock_row = df.get(stock)
                # latest_price = float(stock_row['open'])
                # print(2222222222222, stock, stock_row)
                open_price = float(stock_row['open'])
                latest_price = float(stock_row['lastPrice'])

                # print(code, latest_price, stock_row)
                #schema:名称,最新价,涨跌幅,涨跌额,成交量,成交额,振幅,最高,最低,今开,昨收,量比,换手率,市盈率-动态,市净率,总市值,流通市值,涨速,5分钟涨跌,60日涨跌幅,年初至今涨跌幅
            except IndexError:
                print(f"代码 {code} 未在实时数据中找到，跳过处理")
                continue
            except KeyError:
                print(f"实时数据中缺少必要字段，请检查数据源")
                continue
            except Exception as e:
                print(f"异常：{e}")
                continue
            if latest_price is None or latest_price == 0:
                continue
            last_close_list.append(latest_price)
            if latest_price == 0 or len(last_close_list) < 35:
                continue
            # 利用最新价格计算当天的MA5，MA10，MA20，MA30
            lc_list = last_close_list
            len_lc_list = len(lc_list)
            ma5 = sum(lc_list[len_lc_list - 5:]) / 5.0
            prev_ma5 = info_dict.get("MA5", 1000)
            if prev_ma5 == 1000:
                continue
            ma10 = sum(lc_list[len_lc_list - 10:]) / 10.0
            prev_ma10 = info_dict.get("MA10", 1000)
            ma20 = sum(lc_list[len_lc_list - 20:]) / 20.0
            prev_ma20 = info_dict.get("MA20", 1000)
            ma30 = sum(lc_list[len_lc_list - 30:]) / 30.0
            prev_ma30 = info_dict.get("MA30", 1000)
            ## 今天 ma5 不能最低
            if ma10 > ma5 and ma20 > ma5 and ma30 > ma5:
                continue
            # if code == "600705":
            #     print(f"latest_price={latest_price};ma5={ma5};ma10={ma10};ma20={ma20};ma30={ma30}")

            ## 相对昨天出现的幅度
            # score = (ma5 - prev_ma5) / prev_ma5
            # print(f"[DEBUG]rma industry={industry},code={code}, ma5_inc_rate={score}")
            ma5_rate = (ma5 - prev_ma5) / prev_ma5
            ma10_rate = (ma10 - prev_ma10) / prev_ma10
            ma20_rate = (ma20 - prev_ma20) / prev_ma20
            ma30_rate = (ma30 - prev_ma30) / prev_ma30
            # score = ma5_rate + ma10_rate + ma20_rate + ma30_rate
            score = ma5_rate
            print(f"[DEBUG]rma industry={industry},code={code},score={score}, ma5_rate={ma5_rate}, ma10_rate={ma10_rate}, ma20_rate={ma20_rate}, ma30_rate={ma30_rate}")

            # if score < 0.005:
            #     continue
            code_score_dict[code] = score

        csd_size = len(code_score_dict)
        if csd_size == 0:
            return json.dumps({"warn": [{"mark": "code_score_dict is None."}]})
        # print(f"[DEBUG] 标的数量：{len(code_score_dict)}")
        ## 2.排序
        sorted_codes = sorted(code_score_dict.items(), key=lambda x: x[1], reverse=True)
        # 取前5名（优化筛选逻辑）
        TOP_N = 300
        eff_code_dict = {code: score for code, score in sorted_codes[:TOP_N]}
        # print(f"[DEBUG] 有效标的数量：{len(eff_code_dict)}")
        # print("Top 5 评分结果：")
        ## 3. 是否买入
        target_ind = 0
        for rank, (code, score) in enumerate(eff_code_dict.items(), 1):
            cc = code.split('.')[0]
            # 已经买入的数量是否达到所需补充的数量
            if target_ind == target_num:
                break
            info_dict = self.stock_dict.get(cc, {})
            stock = info_dict.get("code", "")
            industry = info_dict.get("industry", "")
            print(f"第{rank}名 | 行业:{industry} | 代码：{code} | 分数：{score:.4f}")

            last_close_list = [float(ele) for ele in info_dict.get("last_close", "").split(",") if ele]
            latest_price = 0.0
            open_price = 0.0
            max_price = 0.0
            # 从实时数据获取最新价格
            try:
                # 注意：需要确认代码格式是否匹配（如是否包含交易所后缀）
                # stock_row = df[df['代码'] == code].iloc[0]  # 假设代码精确匹配
                # open_price = float(stock_row['今开'])
                # max_price = float(stock_row['最高'])
                # latest_price = float(stock_row['最新价'])
                stock_row = df.get(stock)
                open_price = float(stock_row['open'])
                max_price = float(stock_row['high'])
                # latest_price = float(stock_row['open'])
                latest_price = float(stock_row['lastPrice'])

                # print(code, latest_price, stock_row)
            except IndexError:
                # print(f"代码 {code} 未在实时数据中找到，跳过处理")
                continue
            except KeyError:
                # print(f"实时数据中缺少必要字段，请检查数据源")
                continue
            except Exception as e:
                # print(f"异常：{e}")
                continue
            if latest_price is None or latest_price == 0:
                continue

            # 保留最近5次的股价数据
            if stock not in self.buy_price_history:
                self.buy_price_history[stock] = list()
            self.buy_price_history[stock].append(latest_price)
            if len(self.buy_price_history[stock]) > 5:
                self.buy_price_history[stock].pop(0)

            max_price_timestamp = None
            # 记录最高价时间戳
            if latest_price <= max_price and max_price_timestamp is None:
                max_price_timestamp = datetime.datetime.now()

            last_1d_close_price = last_close_list[-1]
            # print(f"[DEBUG]should_buy_param,stock={stock},latest_price={latest_price},last_1d_close_price={last_1d_close_price}")
            is_buy, buy_id= self.should_buy(stock, latest_price, last_1d_close_price, info_dict, is_high_fx, is_noon_time, max_price)
            print(f"[DEBUG]should_buy,is_buy={is_buy},buy_id={buy_id}")
            if not is_buy:
                continue
            ## 均衡仓位
            buy_volume = self.get_buy_volume(latest_price)
            if buy_volume is None or buy_volume == 0:
                continue

            ## 账户余额足够买入
            if cash >= latest_price * buy_volume:
                ## 避免一天同一只入多次
                if self.wt_dict.get(stock, 0) > 0 or stock_wt_map.get(stock, 0) > 0:
                    continue
                order_id = -1

                if "1" != self.is_test:
                    order_id = xt_trader.order_stock(acc, stock, xtconstant.STOCK_BUY, buy_volume,
                                                     xtconstant.FIX_PRICE, latest_price, strategy_name="rma")
                    target_ind += 1
                self.wt_dict[stock] = 1
                ret = dict()
                ret['code'] = stock
                ret['price'] = latest_price
                ret['action'] = 'buy'
                ret['volume'] = buy_volume
                ret['order_id'] = order_id
                ret['buy_id'] = buy_id
                rma_logger.info("b:" + json.dumps(ret, ensure_ascii=False))
                if "1" == self.is_test:
                    print("b:" + json.dumps(ret, ensure_ascii=False))
                return json.dumps(ret, ensure_ascii=False)



        # print(f"code={code};info_dict={type(df)}")
            # print(f"rt_info={df}")
        # print(data_list)
        #xtdata.run()
        # 返回整合后的结果
        return json.dumps({"msg": ""})

def get_yesterday_date():
    """ 获取大A前一个交易日的日期 """
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    # print(today)
    previous_trade_date = a_trade_calendar.get_pre_trade_date(today).replace('-', '')
    # print(previous_trade_date)
    #return (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y%m%d')
    return previous_trade_date

if __name__ == "__main__":
    a = ReverseMABULLV2(config="../conf/v1.ini")