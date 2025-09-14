#coding=gbk
import os
import time
import glob
import json
import asyncio
import datetime
import a_trade_calendar
import random
from xtquant import xtdata
from xtquant import xtconstant
from base_strategy import BaseStrategy
from utils import utils

"""
todo:
1, 理想涨停量能公式：
理想量比 = 当日volume / 前5日平均volume

量比区间	信号类型	操作建议
<0.5	缩量板	高成功概率
0.5-1.2	健康量	持筹待涨
>3.0	爆量板	次日易分歧
2, 主力资金识别
筛选条件：
成交额 > 5亿元（避免小盘股流动性陷阱）
大单净流入占比 > 20%
资金流向公式：
主力净买额 = 超大单金额 + 大单金额

3， 洗盘强度判断
优质涨停特征：

日内振幅 < 5%（秒板或一字板最佳）

涨停后振幅 < 2%（筹码稳定）
风险信号：
if 振幅 > 15% and 涨停板数 > 3: 警惕天地板

4， 筹码结构分析
换手率区间	市场含义
<5%	锁仓明显
5%-15%	健康换手
>25%	死亡换手
动态阈值计算：
合理换手率 = 流通股本中活跃筹码比例 × 1.5
"""


class Dragon_V5(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.is_test = False
        self.sell_price_history = {}
        self.buy_price_history = {}
        self.industry_num_map = dict()
        self.sorted_gpzt_pools = list()

        ## 加载连板强度指标
        strength_file = self.get_latest_file("./logs", "v1_daily_limit_strength_data")
        self.strenth_dict = json.load(open(strength_file))

        ## 加载候选池
        # pools_file = './logs/dragon_v5_data.20241130'
        pools_file = self.get_latest_file('./logs', 'dragon_v5_data')
        self.stock_dict = json.load(open(pools_file))

        self.pools_list = list()
        keys_list = list(self.stock_dict.keys())
        random.shuffle(keys_list)
        for kk in keys_list:
            # print(kk)
            idict = self.stock_dict.get(kk, {})
            cc = idict.get("code", "")
            is_target = idict.get("is_target", "")
            is_up_limit_before_half_year = idict.get("is_up_limit_before_half_year", "0")
            continuous_up_limit_days = idict.get("continuous_up_limit_days", 0)
            if len(cc) > 0 and is_up_limit_before_half_year == "1" and len(self.pools_list)<60 and continuous_up_limit_days > 0:
                self.pools_list.append(kk)
        print('[dragon_v5][INIT]load pools file name:', pools_file)
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
            print(base_name)
            date_str = base_name.split('_')[-1].split('.')[0]  # 分割出 YYYYMMDD
            return int(date_str)  # 转换为整数用于比较

        # 按日期降序排序后取最新文件
        files_sorted = sorted(files, key=extract_date, reverse=True)
        latest_file = files_sorted[0]
        # latest_file = max(files, key=lambda x: x.split('.')[-1])  # 按日期部分比较文件名
        print(f"[INIT]latest_file={latest_file}")
        return latest_file

    def get_buy_volume(self, current_price, limit_up_days):
        """根据当前股价和连扳数确定买入股数"""
        """
                    根据股票价格和目标总金额，计算应购买的股票数量（按整手计算，1手=100股）
                    :param price: 股票价格（1.0~30.0元）
                    :param target_amount: 目标总金额（如10000元）
                    :return: 股票数量（整百股数）
                """
        target_amount = 3000
        if current_price <= 0 or target_amount <= 0:
            return 0  # 处理非法输入
        # 计算理想手数（可能含小数）
        ideal_hands = target_amount / (current_price * 100)
        # 获取候选手数（地板值和天花板值）
        floor_hands = int(ideal_hands)
        ceil_hands = floor_hands + 1

        # 计算两种手数的实际总金额
        amount_floor = floor_hands * current_price * 100
        amount_ceil = ceil_hands * current_price * 100

        # 比较哪个更接近目标金额（优先选择不超支的方案）
        diff_floor = abs(target_amount - amount_floor)
        diff_ceil = abs(target_amount - amount_ceil)

        # 如果差距相等，优先选择金额较小的方案（如9000 vs 11000时选9000）
        if diff_floor <= diff_ceil:
            return floor_hands * 100
        else:
            return ceil_hands * 100

        # if limit_up_days < 5:
        #     if 0 < current_price <= 5.0:
        #         return 600
        #     elif 5.0 < current_price <= 8.0:
        #         return 500
        #     elif 8.0 < current_price <= 10.0:
        #         return 400
        #     elif current_price > 10.0:
        #         return 300
        # else:
        #     return 100
        # return 0

    def should_sell(self, stock, last_1d_close_price, max_price, max_price_timestamp, last_price):
        """判断是否满足卖出条件"""
        # 卖出情况1：若当天max_price>0.095后超过5分钟股价不再>0.095则卖出
        if max_price_timestamp is not None and max_price is not None:
            limit_up_price = round(last_1d_close_price * 1.095, 2)
            current_time = datetime.datetime.now()
            time_diff = (current_time - max_price_timestamp).total_seconds()
            if time_diff > 300 and max_price >= limit_up_price > last_price:
                return True
        # 卖出情况2：检查最近5次股价是否全部低于前一天收盘价的2%
        if stock not in self.sell_price_history or len(self.sell_price_history[stock]) < 5:
            return False  # 数据不足

        his_price_list = self.sell_price_history[stock]
        lt_target_num = 0
        if len(his_price_list) > 5:
            his_price_list = his_price_list[:5]
        for price in his_price_list:
            if ( price - last_1d_close_price ) / last_1d_close_price < 0.04:
                lt_target_num += 1
        if lt_target_num == 5:
            return True
        else:
            return False

    def should_buy(self, stock_code, current_price, last_1d_close_price, limit_up_days):
        ## 大盘连板强度太弱不开仓
        score = self.strenth_dict.get("score", 0.0)
        zha_rate = self.strenth_dict.get("zha_rate", 0.0)
        all_limit_up_num = self.strenth_dict.get("all_limit_up_num", 0.0)
        if score > 0 and zha_rate > 0 and all_limit_up_num > 0:
            if (zha_rate > 0.59 or all_limit_up_num < 30) and score < 1.6:
                return False

        ## 开盘价比昨日收盘价低于4%则不再买入
        if limit_up_days == 1 and (current_price - last_1d_close_price) / last_1d_close_price < 0.083:
            return False
        elif limit_up_days == 2 and (current_price - last_1d_close_price) / last_1d_close_price < 0.066:
            return False
        elif (current_price - last_1d_close_price) / last_1d_close_price < 0.04:
            return False
        ## 记录了最近5次的历史价格数据
        his_price_list = self.buy_price_history[stock_code]
        if len(his_price_list) > 5:
            his_price_list = his_price_list[:5]
        # 价格趋势判断：避免在下跌趋势中买入
        if any(price > current_price for price in his_price_list):
            return False
        # 连续满足条件的判断，确保股票价格有上涨的连续性，避免在趋势不明时买入
        if len([price for price in his_price_list if price < current_price]) < 3:
            return False
        return True

    def price_update_callback(self, data, xt_trader, acc, pools_list, is_jj_time, is_open_time):
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

                code = wt_info.stock_code.split('.')[0]
                industry = self.stock_dict.get(code, {}).get("industry", "")
                if industry in self.industry_num_map:
                    self.industry_num_map[industry] += 1
                else:
                    self.industry_num_map[industry] = 1

        # 查询持仓股票
        has_stock_obj = xt_trader.query_stock_positions(acc)
        has_stock_list = list()
        for has_stock in has_stock_obj:
            # print('持仓总市值=market_value=', has_stock.market_value)
            # print('成本=open_price=', has_stock.open_price)
            has_volume = has_stock.volume
            has_stock_code = has_stock.stock_code
            if stock_wt_map.get(has_stock_code, 0) == 1:
                continue
            has_stock_list.append(has_stock_code)
            if has_stock_code not in data:
                # print(f"[ERROR]has_stock_code not in data,has_stock_code={has_stock_code}")
                continue
            # print(f"wt about stock_code={data[has_stock_code]}")

            last_price = round(data[has_stock_code]['lastPrice'], 2)
            last_1d_close_price = round(data[has_stock_code]['lastClose'], 2)
            max_price = round(data[has_stock_code]['high'], 2)
            max_price_timestamp = None
            # 记录最高价时间戳
            if last_price == max_price and max_price_timestamp is None:
                max_price_timestamp = datetime.datetime.now()
            # last_1d_close_price = utils.get_close_price(has_stock_code, last_n=1)
            # last_price = utils.get_latest_price(has_stock_code, is_download=True)
            print(f"[DEBUG]sell_before,has_volume={has_volume}, last_price={last_price}, last_1d_close_price={last_1d_close_price}, has_stock_code={has_stock_code}")

            # 保留最近5次的股价数据
            if has_stock_code not in self.sell_price_history:
                self.sell_price_history[has_stock_code] = list()
            self.sell_price_history[has_stock_code].append(last_price)
            if len(self.sell_price_history[has_stock_code]) > 5:
                self.sell_price_history[has_stock_code].pop(0)

            # if has_volume > 0 and (last_price - last_1d_close_price) / last_1d_close_price < 0.02:
            if has_volume > 0 and self.should_sell(has_stock_code, last_1d_close_price, max_price, max_price_timestamp, last_price):
                # 为了避免无法出逃，价格笼子限制，卖出价格不能低于当前价格的98%
                sell_price = round(last_price * 0.99, 2)
                if sell_price < round(last_1d_close_price - last_1d_close_price * 0.1, 2):
                    sell_price = round(last_1d_close_price - last_1d_close_price * 0.1, 2)
                order_id = xt_trader.order_stock(acc, has_stock_code, xtconstant.STOCK_SELL, has_volume,
                                                 xtconstant.FIX_PRICE, sell_price)
                sell = dict()
                sell['code'] = has_stock_code
                sell['price'] = sell_price
                sell['action'] = 'sell'
                sell['order_id'] = order_id
                sell['volume'] = has_volume
                sell_list.append(sell)

        ## 买入委托
        buy_list = list()
        has_stock_num = len(set(has_stock_list))
        if has_stock_num < 6:
            # 查询账户余额
            acc_info = xt_trader.query_stock_asset(acc)
            cash = 0
            if acc_info is not None:
                cash = acc_info.cash
            else:
                return json.dumps({"warn": [{"mark": "cash is None."}]})
            # for stock_code, limit_up_days, yesterday_volume, bidVol, askVol, bidPrice, askPrice \
            #         in pools_list:
            for code in pools_list:
                # print('====', code)
                info_dict = self.stock_dict.get(code.split('.')[0], {})
                stock_code = info_dict.get("code", "")
                limit_up_days = info_dict.get("continuous_up_limit_days", 0)
                industry = info_dict.get("industry", "")
                # print(f'-------{code}==={limit_up_days}==={stock_code}==={info_dict}')
                ## 已经持仓，则不再考虑买入
                if stock_code in has_stock_list:
                    # print("[DEBUG]buy has_stock_code=", stock_code)
                    continue
                ## 当前是否有委托
                if stock_wt_map.get(stock_code, 0) == 1:
                    continue
                ## 一个行业最多持仓一支
                if industry in self.industry_num_map:
                    continue

                ## 当前价格，昨日收盘价格
                # current_price = utils.get_latest_price(stock_code, True)
                # last_1d_close_price = utils.get_close_price(stock_code, last_n=1)
                ## 没有当前tick数据
                if stock_code not in data:
                    # print(f"[ERROR]buy stock_code not in data,stock_code={stock_code}")
                    continue

                # print(f"buy about stock_code={data[stock_code]}")
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
                if not is_jj_time and len(self.buy_price_history[stock_code]) > 5:
                    self.buy_price_history[stock_code].pop(0)
                if is_jj_time:
                    ## 竞价时间不执行买入
                    continue
                if is_open_time:
                    ## 竞价弱势不买入
                    price_list = self.buy_price_history[stock_code]
                    is_dec = is_price_declining(price_list)
                    if is_dec:
                        continue

                ## 是否满足买入条件
                ##   账户余额足够买入：账户余额> 股票价格*100
                ##   当前委托单中不存在此股
                is_buy = self.should_buy(stock_code, current_price, last_1d_close_price, limit_up_days)
                if not is_buy:
                    continue
                ## 均衡仓位:
                ##   若股价<5.0,则最多允许买入400；
                ##   若8.0>=股价>5.0最多允许买入300；
                ##   若10.0>=股价>8.0,最多允许买入200;
                ##   若股价>10.0,最多允许买入100；
                buy_volume = self.get_buy_volume(current_price, limit_up_days)
                if buy_volume is None or buy_volume == 0:
                    continue
                ## 账户余额足够买入
                if cash >= current_price * buy_volume:
                    order_id = xt_trader.order_stock(acc, stock_code, xtconstant.STOCK_BUY, buy_volume,
                                                     xtconstant.FIX_PRICE, current_price)

                    ret = dict()
                    ret['code'] = stock_code
                    ret['price'] = current_price
                    ret['action'] = 'buy'
                    ret['volume'] = buy_volume
                    ret['order_id'] = order_id
                    buy_list.append(ret)
        ret_list = buy_list + sell_list
        return json.dumps({"msg": ret_list})

    async def do(self, accounts):
        print("[DEBUG]do dragon_v5 ", utils.get_current_time(), accounts)
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
        start_time = datetime.datetime.strptime("09:15", "%H:%M").time()
        open_time = datetime.datetime.strptime("09:30", "%H:%M").time()
        mid_start_time = datetime.datetime.strptime("11:30", "%H:%M").time()
        mid_end_time = datetime.datetime.strptime("13:01", "%H:%M").time()
        end_time = datetime.datetime.strptime("14:55", "%H:%M").time()
        is_trade_time = start_time <= cur_time <= mid_start_time or mid_end_time <= cur_time <= end_time
        is_jj_time = jj_start_time <= cur_time <= jj_end_time
        is_open_time = cur_time >= open_time
        ## 不在交易时间不操作
        if not is_trade_time and self.is_test == "0":
            return json.dumps({"warn":[{"mark":"is_not_trade_time."}]})

        # 相同板数成交量最大的作为买入
        eff_stock_list = list()
        limit_1_index = 0
        limit_2_index = 0
        for code in self.pools_list:
            content = self.stock_dict.get(code, {})
            # print('===========', code, content)
            stock_code = content.get("code", "")
            if len(stock_code) == 0:
                continue
            limit_up_days = content.get("continuous_up_limit_days", 0)
            if limit_up_days == 0:
                continue
            if limit_up_days == 1 and limit_1_index < 2:
                limit_1_index += 1
                eff_stock_list.append(stock_code)

            elif limit_up_days == 2 and limit_2_index < 2:
                limit_2_index += 1
                eff_stock_list.append(stock_code)

        ## 最终有效的结果池
        if len(eff_stock_list) == 0:
           return json.dumps({"msg": [{"mark": "eff_stock_list is empty."}]})
        print(f"[DEBUG]eff_stock_size={len(eff_stock_list)};pools_list={eff_stock_list}")

        # 已经持仓股票也需要放到订阅中
        has_stock_obj = xt_trader.query_stock_positions(acc)
        has_stock_list = list()
        has_stock_map = dict()
        for has_stock in has_stock_obj:
            # print('持仓总市值=market_value=', has_stock.market_value)
            # print('成本=open_price=', has_stock.open_price)
            has_volume = has_stock.volume
            has_stock_code = has_stock.stock_code
            has_stock_map[has_stock_code] = has_volume
            has_stock_list.append(has_stock_code)

            code = has_stock_code.split('.')[0]
            industry = self.stock_dict.get(code, {}).get("industry", "")
            if industry in self.industry_num_map:
                self.industry_num_map[industry] += 1
            else:
                self.industry_num_map[industry] = 1
        subscribe_whole_list = list(set(has_stock_list + eff_stock_list))
        print(f"[DEBUG]subscribe_whole_list={subscribe_whole_list}")

        # 注册全推回调函数
        # 这里用一个空的列表来存储返回结果
        final_ret_list = []
        loop = asyncio.get_event_loop()
        # 注册全推回调函数
        def callback(data):
            ret = self.price_update_callback(data, xt_trader, acc, eff_stock_list, is_jj_time, is_open_time)
            if ret is not None:
                final_ret_list.extend(ret)

        xtdata.subscribe_whole_quote(subscribe_whole_list, callback=callback)

        # 非阻塞运行xtdata.run()，例如在后台线程中运行
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, xtdata.run)













        # # 注册全推回调函数
        # # 这里用一个空的列表来存储返回结果
        # final_ret_list = []
        #
        # # 注册全推回调函数
        # def callback(data):
        #     ret = self.price_update_callback(data, xt_trader, acc, eff_stock_list)
        #     if ret is not None:
        #         final_ret_list.extend(ret)
        #
        # xtdata.subscribe_whole_quote(subscribe_whole_list, callback=callback)
        # xtdata.run()
        # 返回整合后的结果
        return json.dumps({"msg": final_ret_list})


def is_highest_bid(self, stock_code):
    """检查该股票的封单量是否是相同连板数中最高"""
    # 获取该股票的连板数
    limit_up_days = self.calculate_limit_up_days(stock_code)

    # 获取相同连板数的所有股票及其封单量
    same_limit_up_stocks = self.get_same_limit_up_stocks(limit_up_days)

    if not same_limit_up_stocks:
        return False  # 没有找到相同连板数的股票

    # 获取该股票的封单量
    current_bid_volume = self.get_current_bid_volume(stock_code)

    # 检查封单量是否为最高
    is_highest = all(current_bid_volume >= volume for _, volume in same_limit_up_stocks)
    return is_highest

def is_price_declining(prices, window=3):
    """检查最近window个时间点是否持续下跌"""
    if len(prices) < window:
        return False
    for i in range(1, window):
        if prices[-i] >= prices[-i-1]:
            return False
    return True

def get_yesterday_date():
    """ 获取大A前一个交易日的日期 """
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    # print(today)
    previous_trade_date = a_trade_calendar.get_pre_trade_date(today).replace('-', '')
    # print(previous_trade_date)
    #return (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y%m%d')
    return previous_trade_date

if __name__ == "__main__":
    a = Dragon_V5(config="../conf/v1.ini")
