# -*- coding: utf-8 -*-

import os
import asyncio
import glob
import json
import random
import datetime
import logging
import a_trade_calendar
from xtquant import xtconstant
from xtquant import xtdata
import threading
from queue import Queue

from base_strategy import BaseStrategy
from utils import utils
current_date = datetime.datetime.now().strftime('%Y-%m-%d')
pp = "F:/tools/dataset/logs"  # 修正这个路径的拼写错误

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

class ReverseMABULL(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        # self.data_queue = Queue()  # 用于接收行情数据的队列
        # self.is_running = True  # 控制线程运行状态

        self.is_test = "0"
        ## 记录买入委托，已经委托订单不再进行重复下单
        self.wt_dict = dict()
        self.sell_price_history = {}
        self.buy_price_history = {}
        self.sorted_gpzt_pools = list()
        pools_file = self.get_latest_file(pp, 'reverse_moving_average_bull_track_')
        self.stock_dict = json.load(open(pools_file))
        self.pools_list = list()
        keys_list = list(self.stock_dict.keys())
        random.shuffle(keys_list)
        # for kk in keys_list:
        for kk, vv in self.stock_dict.items():
            # print(kk)
            idict = self.stock_dict.get(kk, {})
            cc = idict.get("code", "")
            is_target = idict.get("is_target", "")
            if len(cc) > 0 and len(self.pools_list)<60 and is_target == "1":
                self.pools_list.append(cc)
        print('[INIT]load pools file name:', pools_file)
        print('[INIT]load total size:', len(keys_list), keys_list[:5])
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

        # if 0 < current_price <= 2.0:
        #     return 10000
        # elif 2.0 < current_price <= 5.0:
        #     return 2000
        # elif 5.0 < current_price <= 8.0:
        #     return 1500
        # elif 8.0 < current_price <= 10.0:
        #     return 1000
        # elif 10.0 < current_price <= 15.0:
        #     return 100
        # else:
        #     return 100

    def should_sell(self, stock, last_1d_close_price, max_price, max_price_timestamp, last_price, open_price):
        """判断是否满足卖出条件"""
        ## 卖出情况1: 止损位 1 个点
        #o_price = round(open_price * 1.01, 2)
        #if last_price <= o_price:
        #    return True, 1
        ## 卖出情况2: ma5/10平行或向下
        info_dict = self.stock_dict.get(stock, {})
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
            if increase_ma5:
                return True, 6
            ma10 = sum(lc_list[len_lc_list - 10:]) / 10.0
            prev_ma10 = info_dict.get("MA10", 1000)
            # ma10无上升趋势
            increase_ma10 = (ma10 <= prev_ma10)
            if increase_ma10:
                return True, 7

        # 卖出情况3：若当天max_price>0.095后超过5分钟股价不再>0.095则卖出
        if max_price_timestamp is not None and max_price is not None:
            limit_up_price = round(last_1d_close_price * 1.095, 2)
            current_time = datetime.datetime.now()
            time_diff = (current_time - max_price_timestamp).total_seconds()
            if time_diff > 300 and max_price >= limit_up_price > last_price:
                return True, 2
        # 卖出情况4：检查最近3次股价是否全部低于前一天收盘价的1.4%
        if stock not in self.sell_price_history or len(self.sell_price_history[stock]) < 3:
            return False, 3  # 数据不足

        his_price_list = self.sell_price_history[stock]
        lt_target_num = 0
        if len(his_price_list) > 3:
            his_price_list = his_price_list[:3]
        for price in his_price_list:
            if ( price - last_1d_close_price ) / last_1d_close_price < 0.014:
                lt_target_num += 1
        if lt_target_num == 3:
            return True, 4
        else:
            return False, 5

    def should_buy(self, stock_code, current_price, last_1d_close_price, info_dict, is_high_fx, is_noon_time):
        ## 记录了最近5次的历史价格数据
        # his_price_list = self.buy_price_history[stock_code]
        # if len(his_price_list) > 5:
        #     his_price_list = his_price_list[:5]
        # # 价格趋势判断：避免在下跌趋势中买入
        # if any(price > current_price for price in his_price_list):
        #     return False
        # # 连续满足条件的判断，确保股票价格有上涨的连续性，避免在趋势不明时买入
        # if len([price for price in his_price_list if price < current_price]) < 3:
        #     return False
        ## 下午再进行买入操作
        if not is_noon_time:
            return False, 204
        ## 10:25之前不追高，非追高的策略，过高时不再买入，诱多比较多
        # if is_high_fx and ( current_price - last_1d_close_price ) / last_1d_close_price > 0.05:
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
            increase_status = (ma5 > prev_ma5) and (ma10 > prev_ma10) and (ma20 > prev_ma20) and (ma30 > prev_ma30)
            # 多头排列
            bull_track_status = (ma5 > ma10) and (ma10 > ma20) and (ma20 > ma30)
            if increase_status & bull_track_status:
                return True, 202
            else:
                return False, 203


    def price_update_callback(self, data, xt_trader, acc, is_high_fx, is_noon_time):
        '''
          1,查询仓位，若有仓位，判断是否要卖出
          2,仓位小于6支，则继续探索买入
        :param data: 回调数据
        :param xt_trader:
        :param acc:
        :return:
        '''
        #### 卖出判断
        ## 查看是否持仓，若持仓则监控股价是否低于预期，若低于预期则卖出，否则一直持有
        ## 若没有持仓，则监控股价，选择买入
        # 记录卖出日志
        sell_list = list()
        # 查询账户委托
        ## 当前是否有委托单,避免重复报废单
        stock_wt_map = dict()
        wt_infos = xt_trader.query_stock_orders(acc, True)
        for wt_info in wt_infos:
            # print(wt_info.stock_code, wt_info.order_volume, wt_info.price)
            if wt_info.stock_code is not None:
                stock_wt_map[wt_info.stock_code] = 1

        # 查询持仓股票
        has_stock_obj = xt_trader.query_stock_positions(acc)
        has_stock_list = list()
        for has_stock in has_stock_obj:
            # print('持仓总市值=market_value=', has_stock.market_value)
            # print('成本=open_price=', has_stock.open_price)
            open_price = has_stock.open_price
            has_volume = has_stock.volume
            has_stock_code = has_stock.stock_code
            if stock_wt_map.get(has_stock_code, 0) == 1:
                continue
            has_stock_list.append(has_stock_code)
            if has_stock_code not in data:
                # print(f"[ERROR]has_stock_code not in data,has_stock_code={has_stock_code}")
                continue
            last_price = round(data[has_stock_code]['lastPrice'], 2)
            last_1d_close_price = round(data[has_stock_code]['lastClose'], 2)
            max_price = round(data[has_stock_code]['high'], 2)
            max_price_timestamp = None
            # 记录最高价时间戳
            if last_price == max_price and max_price_timestamp is None:
                max_price_timestamp = datetime.datetime.now()
            # last_1d_close_price = utils.get_close_price(has_stock_code, last_n=1)
            # last_price = utils.get_latest_price(has_stock_code, is_download=True)
            # print(f"[DEBUG]sell_before,has_volume={has_volume}, last_price={last_price}, last_1d_close_price={last_1d_close_price}, has_stock_code={has_stock_code}")

            # 保留最近5次的股价数据
            if has_stock_code not in self.sell_price_history:
                self.sell_price_history[has_stock_code] = list()
            self.sell_price_history[has_stock_code].append(last_price)
            if len(self.sell_price_history[has_stock_code]) > 5:
                self.sell_price_history[has_stock_code].pop(0)

            is_sell, sell_id = self.should_sell(has_stock_code, last_1d_close_price, max_price, max_price_timestamp, last_price,
                             open_price)
            if has_volume > 0 and is_sell:
                # 为了避免无法出逃，价格笼子限制，卖出价格不能低于当前价格的98%
                sell_price = round(last_price * 0.99, 2)
                if sell_price < round(last_1d_close_price - last_1d_close_price * 0.1, 2):
                    sell_price = round(last_1d_close_price - last_1d_close_price * 0.1, 2)
                order_id = -1
                if "1" != self.is_test:
                    order_id = xt_trader.order_stock(acc, has_stock_code, xtconstant.STOCK_SELL, has_volume,
                                                xtconstant.FIX_PRICE, sell_price)
                sell = dict()
                sell['code'] = has_stock_code
                sell['price'] = sell_price
                sell['action'] = 'sell'
                sell['order_id'] = order_id
                sell['volume'] = has_volume
                sell["sell_id"] = sell_id
                rma_logger.info("s:" + json.dumps(sell, ensure_ascii=False))

                # sell_list.append(sell)

        ## 买入委托
        buy_list = list()
        has_stock_num = len(set(has_stock_list))
        if has_stock_num < 200:
            # 查询账户余额
            acc_info = xt_trader.query_stock_asset(acc)
            cash = acc_info.cash

            for code in self.pools_list:

                info_dict = self.stock_dict.get(code.split('.')[0], {})
                stock_code = info_dict.get("code", "")
                # print(f'-------{code}==={stock_code}==={info_dict}')
                ## 已经持仓，则不再考虑买入
                if stock_code in has_stock_list:
                    # print("[DEBUG]buy has_stock_code=", stock_code)
                    continue
                ## 当前是否有委托
                if stock_wt_map.get(stock_code, 0) == 1:
                    continue
                ## 当前价格，昨日收盘价格
                # current_price = utils.get_latest_price(stock_code, True)
                # last_1d_close_price = utils.get_close_price(stock_code, last_n=1)
                ## 没有当前tick数据
                if stock_code not in data:
                    # print(f"[WARN]buy stock_code not in data,stock_code={stock_code},data_size={len(data.keys())};data={json.dumps(data)}")
                    continue
                # print(
                #     f"[WARN]buy stock_code not in data,stock_code={stock_code},data_size={len(data.keys())};data={json.dumps(data)}")

                current_price = data[stock_code]['lastPrice']
                last_1d_close_price = data[stock_code]['lastClose']
                if current_price is None or last_1d_close_price is None:
                    continue
                if current_price <= 0 or last_1d_close_price <= 0:
                    continue

                # 保留最近5次的股价数据
                if stock_code not in self.buy_price_history:
                    self.buy_price_history[stock_code] = list()
                self.buy_price_history[stock_code].append(current_price)
                if len(self.buy_price_history[stock_code]) > 5:
                    self.buy_price_history[stock_code].pop(0)

                ## 是否满足买入条件
                ##   账户余额足够买入：账户余额> 股票价格*100
                ##   当前委托单中不存在此股
                is_buy, buy_id = self.should_buy(stock_code, current_price, last_1d_close_price, info_dict, is_high_fx, is_noon_time)
                if not is_buy:
                    continue
                ## 均衡仓位:
                ##   若股价<5.0,则最多允许买入400；
                ##   若8.0>=股价>5.0最多允许买入300；
                ##   若10.0>=股价>8.0,最多允许买入200;
                ##   若股价>10.0,最多允许买入100；
                buy_volume = self.get_buy_volume(current_price)
                if buy_volume is None or buy_volume == 0:
                    continue
                ## 账户余额足够买入
                if cash >= current_price * buy_volume:
                    ## 避免一天同一只入多次
                    if self.wt_dict.get(stock_code, 0) == 1:
                        continue
                    order_id = -1
                    if "1" != self.is_test:
                        order_id = xt_trader.order_stock(acc, stock_code, xtconstant.STOCK_BUY, buy_volume,
                                                    xtconstant.FIX_PRICE, current_price, strategy_name="rma")
                    self.wt_dict[stock_code] = 1
                    ret = dict()
                    ret['code'] = stock_code
                    ret['price'] = current_price
                    ret['action'] = 'buy'
                    ret['volume'] = buy_volume
                    ret['order_id'] = order_id
                    ret['buy_id'] = buy_id
                    rma_logger.info("b:"+json.dumps(ret, ensure_ascii=False))
                    buy_list.append(ret)
        ret_list = buy_list + sell_list
        return json.dumps({"msg": ret_list})

    async def do(self, accounts):
        print("[DEBUG]do reverse_moving_average_bull_track ", utils.get_current_time(), accounts)
        target_code = '沪深A股'
        req_dict = accounts.get("acc_1", {})
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
        start_time = datetime.datetime.strptime("09:30", "%H:%M").time()
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

        # 已经持仓股票也需要放到订阅中
        has_stock_obj = xt_trader.query_stock_positions(acc)
        has_stock_list = list()
        has_stock_map = dict()
        stock_list = self.pools_list
        for has_stock in has_stock_obj:
            # print('持仓总市值=market_value=', has_stock.market_value)
            # print('成本=open_price=', has_stock.open_price)
            has_volume = has_stock.volume
            has_stock_code = has_stock.stock_code
            has_stock_map[has_stock_code] = has_volume
            has_stock_list.append(has_stock_code)
        subscribe_whole_list = list(set(has_stock_list + stock_list))
        print(f"[DEBUG]subscribe_whole_list={len(subscribe_whole_list)}={subscribe_whole_list}")

        # # 启动独立线程运行 xtdata
        # def xtdata_thread():
        #     xtdata.run()
        #
        # thread = threading.Thread(target=xtdata_thread, daemon=True)
        # thread.start()
        #
        # # 异步处理数据队列中的回调
        # async def process_data_queue():
        #     while self.is_running:
        #         if not self.data_queue.empty():
        #             data = self.data_queue.get()
        #             ret = self.price_update_callback(data, xt_trader, acc, is_high_fx, is_noon_time)
        #             # 处理返回结果（示例）
        #         await asyncio.sleep(0.1)  # 避免CPU占满
        #
        # # 启动数据处理任务
        # asyncio.create_task(process_data_queue())
        #
        # # 立即返回，不阻塞后续interval
        # return json.dumps({"status": "策略已启动"})

        # # quit()
        # 注册全推回调函数
        # 这里用一个空的列表来存储返回结果
        final_ret_list = []
        loop = asyncio.get_event_loop()
        # 注册全推回调函数
        def callback(data):
            ret = self.price_update_callback(data, xt_trader, acc, is_high_fx, is_noon_time)
            if ret is not None:
                final_ret_list.extend(ret)

        xtdata.subscribe_whole_quote(subscribe_whole_list, callback=callback)

        # 非阻塞运行xtdata.run()，例如在后台线程中运行
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, xtdata.run)
        #
        # # 启动策略逻辑任务，不等待其完成
        # asyncio.create_task(run_strategy_logic())
        # return json.dumps({"status": "策略已异步启动"})





        #xtdata.run()
        # 返回整合后的结果
        # return json.dumps({"msg": final_ret_list})

def get_yesterday_date():
    """ 获取大A前一个交易日的日期 """
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    # print(today)
    previous_trade_date = a_trade_calendar.get_pre_trade_date(today).replace('-', '')
    # print(previous_trade_date)
    #return (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y%m%d')
    return previous_trade_date

if __name__ == "__main__":
    a = ReverseMABULL(config="../conf/v1.ini")