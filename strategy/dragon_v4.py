#coding=gbk
import os
import time
import glob
import json
import datetime
import a_trade_calendar
from mpmath import limit
from xtquant import xtdata
from xtquant import xtconstant
from base_strategy import BaseStrategy
from utils import utils

'''
策略逻辑

条件：
1，前一个交易日涨停的沪深
2，排除ST内容
3，排除单价在2元以下,40元以上
4，开盘价在x%以上内容


买入：
1，每分钟股价在2%以上，买入概率随有效分钟数增加0.01
2，若有大单买入，而增加

出逃：
1，一天最多允许买入一支，行情结束再买下一支？
'''


class Dragon_V4(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.sell_price_history = {}
        self.buy_price_history = {}
        self.sorted_gpzt_pools = list()
        # pools_file = './logs/dragon_v4_data.20241130'
        pools_file = self.get_latest_file('./logs', 'dragon_v4_data.')
        self.load_stock_pools(pools_file)

    def get_latest_file(self, directory, prefix):
        # 获取目录下所有以 'prefix' 开头的文件
        files = glob.glob(os.path.join(directory, f"{prefix}*"))
        # 如果没有找到文件，返回 None
        if not files:
            return None
        # 从文件名中提取日期部分，并找到最新的文件
        latest_file = max(files, key=lambda x: x.split('.')[-1])  # 按日期部分比较文件名
        print(f"[DEBUG]INIT:latest_file={latest_file}")
        return latest_file

    def load_stock_pools(self, pools_file):
        for i, line in enumerate(open(pools_file)):
            line = line.rstrip('\r\n')
            lines = line.split('\t')
            if len(lines) < 3:
                continue
            stock_code = lines[0].strip()
            limit_up_days = int(lines[1].strip())
            yesterday_volume = int(lines[2].strip())
            bidVol = int(lines[3].strip())
            askVol = int(lines[4].strip())
            bidPrice = float(lines[5].strip())
            askPrice = float(lines[6].strip())

            self.sorted_gpzt_pools.append((stock_code, limit_up_days, yesterday_volume, bidVol, askVol, bidPrice, askPrice))

    def get_buy_volume(self, current_price, limit_up_days):
        """根据当前股价和连扳数确定买入股数"""
        if limit_up_days < 5:
            if 0 < current_price <= 5.0:
                return 600
            elif 5.0 < current_price <= 8.0:
                return 500
            elif 8.0 < current_price <= 10.0:
                return 400
            elif current_price > 10.0:
                return 300
        else:
            return 100
        return 0

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

    def price_update_callback(self, data, xt_trader, acc, pools_list):
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
            cash = acc_info.cash
            for stock_code, limit_up_days, yesterday_volume, bidVol, askVol, bidPrice, askPrice \
                    in pools_list:
                # ## 集合竞价时间：检查该股票的封单量是否相同连板数中最高，若不是则取消委托。
                # if jj_start_time <= cur_time <= jj_end_time:
                #     ## 检查该股票的封单量是否相同连板数中最高
                #     is_highest = is_highest_bid(stock_code)
                #     if not is_highest:
                #         action_name = 'cancel'
                #         ## 取消买入委托的订单
                #         xt_trader.cancel_order_stock(acc, order_id)
                ## 已经持仓，则不再考虑买入
                if stock_code in has_stock_list:
                    print("[DEBUG]buy has_stock_code=", stock_code)
                    continue
                ## 当前是否有委托
                if stock_wt_map.get(stock_code, 0) == 1:
                    continue
                ## 当前价格，昨日收盘价格
                # current_price = utils.get_latest_price(stock_code, True)
                # last_1d_close_price = utils.get_close_price(stock_code, last_n=1)
                ## 没有当前tick数据
                if stock_code not in data:
                    print(f"[ERROR]buy stock_code not in data,stock_code={stock_code}")
                    continue
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

    def do(self, accounts):
        print("[DEBUG]do dragon_v4 ", utils.get_current_time(), accounts)
        target_code = '沪深A股'
        req_dict = accounts.get("acc_1", {})
        xt_trader = req_dict.get("xt_trader")
        acc = req_dict.get("account")
        # acc_name = req_dict.get("acc_name")
        ## 加载有效召回池
        sorted_stocks = self.sorted_gpzt_pools
        print(f"[DEBUG]sorted_stocks={sorted_stocks}")
        if len(sorted_stocks) == 0:
            return json.dumps({"msg":[{"mark":"sorted_stocks is empty."}]})

        # 相同板数成交量最大的作为买入
        pools_list = list()
        eff_stock_list = list()
        limit_1_index = 0
        limit_2_index = 0
        limit_3_index = 0
        limit_4_index = 0
        limit_5_index = 0
        for content in sorted_stocks:
            stock_code, limit_up_days, yesterday_volume, bidVol, askVol, bidPrice, askPrice = content
            if bidVol > 10000 and askVol == 0 and limit_up_days == 1:
                pools_list.append(content)
                eff_stock_list.append(stock_code)
                limit_1_index += 1
            elif bidVol > 10000 and askVol == 0 and limit_up_days == 2:  # and limit_2_index < 4:
                pools_list.append(content)
                eff_stock_list.append(stock_code)
                limit_2_index += 1
            elif bidVol > 10000 and askVol == 0 and limit_up_days == 3:
                pools_list.append(content)
                eff_stock_list.append(stock_code)
                limit_3_index += 1

        ## 最终有效的结果池
        if len(pools_list) == 0:
            return json.dumps({"msg":[{"mark":"pools_list is empty."}]})
        print(f"[DEBUG]pools_list_size={len(pools_list)};pools_list={pools_list}")

        ## 辅助时间
        cur_time = datetime.datetime.now().time()
        gy_time = datetime.datetime.strptime("22:01", "%H:%M").time()
        jj_start_time = datetime.datetime.strptime("09:15", "%H:%M").time()
        jj_end_time = datetime.datetime.strptime("09:19", "%H:%M").time()
        start_time = datetime.datetime.strptime("09:31", "%H:%M").time()
        mid_start_time = datetime.datetime.strptime("11:30", "%H:%M").time()
        mid_end_time = datetime.datetime.strptime("13:01", "%H:%M").time()
        end_time = datetime.datetime.strptime("14:55", "%H:%M").time()
        is_trade_time = start_time <= cur_time <= mid_start_time or mid_end_time <= cur_time <= end_time
        is_jj_time = jj_start_time <= cur_time <= jj_end_time

        ## 不在交易时间不操作
        if not is_trade_time:
            return json.dumps({"msg":[{"mark":"is_not_trade_time."}]})

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
        subscribe_whole_list = list(set(has_stock_list + eff_stock_list))
        print(f"[DEBUG]subscribe_whole_list={subscribe_whole_list}")
        # 注册全推回调函数
        # 这里用一个空的列表来存储返回结果
        final_ret_list = []

        # 注册全推回调函数
        def callback(data):
            ret = self.price_update_callback(data, xt_trader, acc, pools_list)
            if ret is not None:
                final_ret_list.extend(ret)

        xtdata.subscribe_whole_quote(subscribe_whole_list, callback=callback)
        xtdata.run()
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

def get_yesterday_date():
    """ 获取大A前一个交易日的日期 """
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    # print(today)
    previous_trade_date = a_trade_calendar.get_pre_trade_date(today).replace('-', '')
    # print(previous_trade_date)
    #return (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y%m%d')
    return previous_trade_date