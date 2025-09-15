# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

# 添加项目根目录到sys.path
current_file_path = Path(__file__).resolve()
# 获取项目根目录（当前文件在'项目根目录/strategy'下）
project_root = current_file_path.parent.parent
sys.path.insert(0, str(project_root))

import glob
import json
import datetime
import logging
import a_trade_calendar
import pandas as pd
import numpy as np
import torch

from xtquant import xtconstant
from xtquant import xtdata
from base_strategy import BaseStrategy
from utils import utils
from incubate.train_mlp_mult_label_v2.train_mlp_v1 import MultiClassStockModel, RollingNormalizer
current_date = datetime.datetime.now().strftime('%Y-%m-%d')
pp = "D:/tool/dataset/logs"

# 检查并创建日志目录
if not os.path.exists(pp):
    os.makedirs(pp, exist_ok=True)

log_file = os.path.join(pp, f"mlp_infer_v2_{current_date}.log")
print(log_file)

# 创建一个新的日志记录器
rma_logger = logging.getLogger("MlpInferV2")
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



class StockModelPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        # 打印设备信息
        if self.device.type == "cuda":
            print(f"使用GPU进行推理: {torch.cuda.get_device_name(0)}")
        else:
            print("使用CPU进行推理")

    def _load_model(self):
        # 加载模型检查点
        checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))

        # 获取模型文件所在目录
        model_dir = os.path.dirname(os.path.abspath(self.model_path))

        # 滚动归一化器 - 使用绝对路径
        # rolling_normalizer = None
        # if 'rolling_normalizer_path' in checkpoint:
        #     rnp = checkpoint['rolling_normalizer_path']
        #     print(f"rolling_normalizer_path={rnp}")
        #     # 确保RollingNormalizer类已导入
        #     rolling_normalizer = RollingNormalizer.load(checkpoint['rolling_normalizer_path'])
        # else:
        #     raise ValueError("检查点中缺少rolling_normalizer_path信息")

        rolling_normalizer_path = checkpoint.get('rolling_normalizer_path')
        if rolling_normalizer_path:
            # 构建绝对路径
            rolling_normalizer_path = os.path.join(model_dir, rolling_normalizer_path)
            if os.path.exists(rolling_normalizer_path):
                rolling_normalizer = RollingNormalizer.load(rolling_normalizer_path)
            else:
                raise FileNotFoundError(f"归一化器文件不存在: {rolling_normalizer_path}")
        else:
            raise ValueError("检查点中缺少rolling_normalizer_path信息")

        # 从检查点获取配置
        config = checkpoint['config']

        # 创建模型实例 - 传递有效的rolling_normalizer
        model = MultiClassStockModel(
            num_classes=config['num_classes'],
            rolling_normalizer=rolling_normalizer  # 传递加载的归一化器
        )

        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def preprocess_input(self, raw_data):
        """将原始输入转换为模型需要的张量格式"""
        features = {}

        # 整数特征
        int_features = ['code', 'year', 'month', 'day', 'hour', 'minute', 'day_of_week']
        for key in int_features:
            if key in raw_data:
                features[key] = torch.tensor([raw_data[key]], dtype=torch.long).to(self.device)

        # 浮点数特征
        float_features = ['open', 'close', 'high', 'low', 'volume',
                          'minute_sin', 'minute_cos', 'hour_sin', 'hour_cos',
                          'day_sin', 'day_cos', 'month_sin', 'month_cos']
        for key in float_features:
            if key in raw_data:
                features[key] = torch.tensor([raw_data[key]], dtype=torch.float32).to(self.device)

        return features

    def predict(self, features):
        """使用模型进行预测"""
        # 准备输入数据
        device_features = {k: v.to(self.device) for k, v in features.items()}

        with torch.no_grad():
            logits = self.model(device_features)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        return preds.cpu().numpy(), probs.cpu().numpy()

class MlpInferV2(BaseStrategy):
    def update_config(self, new_config):
        """动态更新配置参数"""
        self.config = new_config
        # 可添加特定参数的更新逻辑（如调整阈值）

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.is_test = "0"
        ## 记录买入委托，已经委托订单不再进行重复下单
        self.wt_dict = dict()
        self.sell_price_history = {}
        self.buy_price_history = {}
        self.sorted_gpzt_pools = list()

        ## 加载模型文件
        model_file = 'D:/tool/pycharm_workspace/Hermes/strategy/model_mlp_infer_v2/stock_model_finalv2.pth'
        self.predictor = StockModelPredictor(model_file)

        ## 加载code&index对应关系数据，也是标的池
        self.code2index = dict()     # 无后缀，{'600051': '76621'}
        self.pools_list = list()     # 有后缀，'600051.SH'
        stock_index_file = "D:/tool/dataset/lookup_stock_code.json"
        stock_white_file = "D:/tool/dataset/stock_pools.json"
        self._load_code_index(stock_index_file, stock_white_file)
        # print(f"[INIT]code2index_size={len(self.code2index.keys())}，sample={self.code2index}")
        # print(f"[INIT]pools_list={self.pools_list}")


        ## 加载当天推理所需的high，low，volume数据
        self.infer_data = dict()
        infer_data_file = self.get_latest_file('D:/tool/dataset/infer_data', 'hs_quant_base_')
        # self.infer_data = json.load(open(infer_data_file, encoding="utf-8"))
        with open(infer_data_file, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line.rstrip("\r\n"))
                for k, v in d.items():
                    self.infer_data[k] = v
        print(f"[INIT]infer_dict={self.infer_data['000560']}")

        ## 加载卖点依赖的ma数据
        pools_file = self.get_latest_file('D:/tool/dataset/strategy_data', 'rtrma_')
        # pools_file = './logs/reverse_moving_average_bull_track_20250419.json'
        self.stock_dict = json.load(open(pools_file))





        # 创建Excel交易记录文件
        self.trade_record_file = os.path.join(pp, f"trade_record_{current_date}.xlsx")

        # 初始化持仓记录字典
        self.position_records = {}
        print('[INIT]SUCCEED!')

    def _load_code_index(self, stock_index_file, stock_white_file):
        """加载股票代码到索引的映射"""
        cc_list = list()
        with open(stock_white_file, 'r') as f:
            vv = json.load(f)
            for k, v in vv.items():
                kk = k.split('.')[0]
                cc_list.append(kk)

        if os.path.exists(stock_index_file):
            with open(stock_index_file, 'r', encoding='utf-8') as f:
                vv = json.load(f)
                for k, v in vv.items():
                    kk = k.split('.')[0]
                    self.code2index[kk] = v
                    if kk not in cc_list:
                        continue
                    self.pools_list.append(k)
            print(f"[INIT]Loaded stock code index mapping, size: {len(self.code2index)},pools_list size: {len(self.pools_list)}")
        else:
            print(f"[INT]Warning: Stock code lookup file not found: {stock_index_file}, pools_list size: {len(self.pools_list)}")

    def get_holding_days(self, stock_code):
        """计算持仓天数"""
        return super().get_holding_days(stock_code)

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
        target_amount = 2000
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

    def should_sell(self, stock, last_1d_close_price, max_price, max_price_timestamp, last_price, bid_price, bid_vol, has_open_price, open_price):
        """判断是否满足卖出条件"""
        # 卖：下午14:50以后价格<开盘价
        cur_time = datetime.datetime.now().time()
        # if cur_time > datetime.time(14, 45) and open_price > last_price:
        #     return True, -3
        ## 卖出情况0：止盈位,一天内利润8%
        # if (last_price - last_1d_close_price) / last_1d_close_price > 0.08:
        #     return True, 0
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
        # 卖出条件：持仓超过3天且收益低于6 %
        if has_open_price > 0:
            # 计算持仓天数
            holding_days = self.get_holding_days(stock)

            # 计算当前收益率
            current_return = (last_price - has_open_price) / has_open_price

            # 持仓超过3天且收益低于6%
            if holding_days >= 3 and current_return < 0.06:
                return True, 10

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

    def should_buy(self, stock_code, current_price, last_1d_close_price, info_dict, is_high_fx, is_noon_time):
        # 下午14:00
        cur_time = datetime.datetime.now().time()
        if cur_time < datetime.time(14, 00):
            return False, 198
        
        
        ## 当天涨幅>x%不再追高，即使买入溢价不高，反而可能诱多跌幅较大
        # if (current_price - last_1d_close_price ) / last_1d_close_price > 0.067:
        #     return False, 199

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

            # increase_status = ma5_rate >= 0.005 #and ma10_rate > 0.0025 and ma20_rate > 0.001 and ma30_rate >= 0
            increase_status = ma5_rate >= 0.005
            # ma5非最低
            is_ma5_not_lowest = ma5 >= ma10 or ma5 >= ma20 or ma5 >= ma30
            # is_ma5_not_lowest = ma5 > ma10 or ma5 > ma20 or ma5 > ma30
            # print(f"stock={stock_code},ma5_rate={ma5_rate}")
            # return True, 201
            if increase_status:
                return True, 202
            else:
                return False, 203

    def get_model_prediction(self, code, features):
        """获取模型预测结果"""
        date = datetime.datetime.now()
        cur_hour = date.hour
        cur_minute = date.minute
        code_index = self.code2index.get(code.split('.')[0], -1)
        if code_index == -1:
            print(f"Warning: No index found for stock code {code}")

        infer_data = self.infer_data[code.split('.')[0]]
        op = features['open']
        cl = features['close']
        high = features['high']
        low = features['low']
        volume = features['volume']
        if infer_data is not None:
            minute_data = infer_data['d']
            # print(minute_data)
            if len(minute_data) > 21:
                volume = sum([int(ele) for ele in minute_data[5]])
                high = max([float(ele) for ele in minute_data[3]])
                low = min([float(ele) for ele in minute_data[4]])

        # 输入特征
        raw_input = {
            'open': op,
            'close': cl,
            'high': high,
            'low': low,
            'volume': volume,
            'code': int(code_index),
            'year': date.year,
            'month': date.month,
            'day': date.day,
            'hour': cur_hour,  # 假设下午2点交易
            'minute': cur_minute,  # 假设下午2:30
            'day_of_week': date.weekday(),
            # 周期特征
            'minute_sin': np.sin(2 * np.pi * cur_minute / 60),
            'minute_cos': np.cos(2 * np.pi * cur_minute / 60),
            'hour_sin': np.sin(2 * np.pi * cur_hour / 24),
            'hour_cos': np.cos(2 * np.pi * cur_hour / 24),
            'day_sin': np.sin(2 * np.pi * date.day / 31),
            'day_cos': np.cos(2 * np.pi * date.day / 31),
            'month_sin': np.sin(2 * np.pi * date.month / 12),
            'month_cos': np.cos(2 * np.pi * date.month / 12)
        }

        # 使用模型进行预测
        processed_input = self.predictor.preprocess_input(raw_input)
        _, probs = self.predictor.predict(processed_input)

        # 提取预测概率
        probabilities = probs[0]  # 形状为 (num_classes,)
        sorted_indices = np.argsort(probabilities)[::-1]  # 从大到小排序

        # 保存预测结果
        prediction = {
            'probabilities': probabilities,
            'top_class': sorted_indices[0],
            'top_prob': probabilities[sorted_indices[0]]
        }
        return prediction


    async def do(self, accounts):
        print("[DEBUG]do mlp infer v2", utils.get_current_time(), accounts)
        target_code = '沪深A股'
        # req_dict = accounts.get("acc_1", {})
        req_dict = accounts.get("acc_2", {})
        xt_trader = req_dict.get("xt_trader")
        acc = req_dict.get("account")
        is_test = req_dict.get("is_test", "0")
        self.is_test = is_test
        # acc_name = req_dict.get("acc_name")
        # 获取账户信息用于记录
        account_name = req_dict.get("acc_name", "未知")

        ## 加载有效召回池
        if len(self.pools_list) == 0:
            return json.dumps({"warn":[{"mark":"pools_list is empty."}]})

        ## 辅助时间
        # cur_time = datetime.datetime.now().time()
        now = datetime.datetime.now()
        cur_time = now.time().replace(microsecond=0)
        gy_time = datetime.datetime.strptime("22:01", "%H:%M").time()
        jj_start_time = datetime.datetime.strptime("09:15", "%H:%M").time()
        jj_end_time = datetime.datetime.strptime("09:19", "%H:%M").time()
        start_time = datetime.datetime.strptime("09:32", "%H:%M").time()
        # start_time = datetime.datetime.strptime("10:32", "%H:%M").time()
        mid_start_time = datetime.datetime.strptime("11:30", "%H:%M").time()
        mid_end_time = datetime.datetime.strptime("13:01", "%H:%M").time()
        end_time = datetime.datetime.strptime("14:55", "%H:%M").time()
        high_fx_time = datetime.datetime.strptime("10:25", "%H:%M").time()
        after_noon_time = datetime.datetime.strptime("14:00", "%H:%M").time()
        is_high_fx = cur_time <= high_fx_time
        is_noon_time = cur_time > after_noon_time
        is_trade_time = start_time <= cur_time <= mid_start_time or mid_end_time <= cur_time <= end_time
        # is_trade_time = mid_end_time <= cur_time <= end_time
        is_jj_time = jj_start_time <= cur_time <= jj_end_time

        ## 不在交易时间不操作
        if (not is_trade_time) and self.is_test == "0":
            print(f'not is_trade_time={is_trade_time}, cur_time={cur_time},start_time={start_time},mid_end_time={mid_end_time}')
            return json.dumps({"warn":[{"mark":"is_not_trade_time."}]})
        print(f"is_trade_time={is_trade_time},is_test={is_test}")

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
        ## 已经持仓+标的数据合并，查询最新行情数据，需要都有后缀
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
            # 只卖出自己策略买入的股票，这里需要修改，暂时保持这样
            if self.get_holding_volume(stock) <= 0 and has_volume <= 0:
                continue
            # print("___________sell_____________", code)
            latest_price = 0.0
            open_price = 0.0
            high_price = 0.0
            low_price = 0.0
            lastClose = 0.0
            bid_price = [0, 0, 0, 0, 0]
            bid_vol = [0, 0, 0, 0, 0]
            volume = 0
            # 从实时数据获取最新价格
            try:
                stock_row = df.get(stock)
                # print(stock_row)
                # {'time': 1755241204000, 'timetag': '20250815 15:00:04', 'lastPrice': 11.93, 'open': 11.78, 'high': 12.09, 'low': 11.56, 'lastClose': 11.82, 'amount': 234458200, 'volume': 198242, 'pvolume': 19824156, 'stockStatus': 5, 'openInt': 15, 'settlementPrice': 0, 'lastSettlementPrice': 11.82, 'askPrice': [11.93, 11.94, 11.95, 11.96, 11.97], 'bidPrice': [11.92, 11.91, 11.9, 11.89, 11.88], 'askVol': [1108, 403, 201, 113, 87], 'bidVol': [420, 247, 760, 66, 222]}
                open_price = float(stock_row['open'])
                high_price = float(stock_row['high'])
                low_price = float(stock_row['low'])
                latest_price = float(stock_row['lastPrice'])
                volume = float(stock_row['volume'])
                lastClose = float(stock_row['lastClose'])
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

            ## 拼接特征
            features = dict()
            features['open'] = open_price
            features['close'] = latest_price
            features['high'] = high_price
            features['low'] = low_price
            features['volume'] = volume
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
            if latest_price <= high_price and max_price_timestamp is None:
                max_price_timestamp = datetime.datetime.now()

            # is_sell = False
            # if pred < 11 or (latest_price - lastClose) / lastClose < -0.059:
            #     is_sell = True
            is_sell, sell_id = self.should_sell(stock, lastClose, high_price, max_price_timestamp, latest_price, bid_price, bid_vol, has_open_price, open_price)

            if is_sell and has_volume > 0:
                ## 上午不再卖出，尊重模型昨日的预估
                is_permit_sell_time = False
                cur_time = datetime.datetime.now().time()
                if (latest_price - lastClose) / lastClose < -0.059 or datetime.time(14, 45) < cur_time < datetime.time(14, 57):
                    is_permit_sell_time = True

                # 为了避免无法出逃，价格笼子限制，卖出价格不能低于当前价格的98%
                sell_price = round(latest_price * 0.99, 2)
                if sell_price < round(lastClose - lastClose * 0.1, 2):
                    sell_price = round(lastClose - lastClose * 0.1, 2)

                order_id = -1
                sell = dict()
                sell['code'] = stock
                sell['price'] = sell_price
                # sell['pred'] = pred
                # sell['pred_score'] = pred_score
                sell['action'] = 'sell'
                sell['volume'] = has_volume
                stock_name = self.stock_dict.get(stock.split('.')[0], {}).get("name", "未知")
                sell['name'] = stock_name
                if "1" != self.is_test:
                    ## 线上执行卖出时的真实订单id，更新持仓记录
                    if is_permit_sell_time:
                        order_id = xt_trader.order_stock(acc, stock, xtconstant.STOCK_SELL, has_volume,
                                                xtconstant.FIX_PRICE, sell_price)
                        sell['order_id'] = order_id

                        ## 更新持仓
                        # stock_name = stock
                        self.record_trade(
                            action='sell',
                            stock_code=stock,
                            stock_name=stock_name,
                            price=sell_price,
                            volume=has_volume,
                            order_id=order_id,
                            reason=f"S{sell_id}",
                            account=account_name
                        )
                else:
                    print(f"[DEBUG]is_test=1,sell_stock={stock},sell_price={sell_price}，sell_volume={has_volume},sell_id={sell_id},name={stock_name},account={account_name}")

                rma_logger.info("s:" + json.dumps(sell, ensure_ascii=False))
                if "1" == self.is_test:
                    print("s:" + json.dumps(sell, ensure_ascii=False))

        ## 动态判断是否要开仓
        # min_hold = 3    # 最小持仓
        # max_hold = 6   # 最大持仓

        min_hold = 10     # 最小持仓
        max_hold = 30   # 最大持仓
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
            latest_price = 0.0
            open_price = 0.0
            high_price = 0.0
            low_price = 0.0
            lastClose = 0.0
            bid_price = [0, 0, 0, 0, 0]
            bid_vol = [0, 0, 0, 0, 0]
            volume = 0
            # 从实时数据获取最新价格
            try:
                stock_row = df.get(code)
                # print(stock_row)
                # {'time': 1755241204000, 'timetag': '20250815 15:00:04', 'lastPrice': 11.93, 'open': 11.78, 'high': 12.09, 'low': 11.56, 'lastClose': 11.82, 'amount': 234458200, 'volume': 198242, 'pvolume': 19824156, 'stockStatus': 5, 'openInt': 15, 'settlementPrice': 0, 'lastSettlementPrice': 11.82, 'askPrice': [11.93, 11.94, 11.95, 11.96, 11.97], 'bidPrice': [11.92, 11.91, 11.9, 11.89, 11.88], 'askVol': [1108, 403, 201, 113, 87], 'bidVol': [420, 247, 760, 66, 222]}
                open_price = float(stock_row['open'])
                high_price = float(stock_row['high'])
                low_price = float(stock_row['low'])
                latest_price = float(stock_row['lastPrice'])
                volume = float(stock_row['volume'])
                lastClose = float(stock_row['lastClose'])
                bid_price = stock_row['bidPrice']
                bid_vol = stock_row['bidVol']

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

            ## 拼接特征
            features = dict()
            features['open'] = open_price
            features['close'] = latest_price
            features['high'] = high_price
            features['low'] = low_price
            features['volume'] = volume
            # 推理，后期改成batch
            prediction = self.get_model_prediction(code, features)
            pred = int(prediction['top_class'])
            pred_score = float(prediction['top_prob'])

            code_score_dict[code] = (pred, pred_score)

        csd_size = len(code_score_dict)
        if csd_size == 0:
            return json.dumps({"warn": [{"mark": "code_score_dict is None."}]})
        # print(f"[DEBUG] 标的数量：{len(code_score_dict)}")
        ## 2.排序
        # 按 pred_score 降序排序（元组的第二个元素）
        sorted_codes = sorted(code_score_dict.items(), key=lambda x: x[1][1], reverse=True)
        # print(json.dumps(sorted_codes))
        # 转换为字典结构
        result_dict = {}
        log_dict = {}
        # 当天预估最大类别
        max_pred = max([int(pred) for code, (pred, info) in code_score_dict.items()])
        # 当天最大类别的股票数量
        max_pred_num = len([pred for code, (pred, info) in code_score_dict.items() if int(pred) == max_pred])
        # 当天第2类别
        second_pred = max([int(pred) for code, (pred, info) in code_score_dict.items() if int(pred) < max_pred])
        # 当天第2类别数量
        second_pred_num = len([pred for code, (pred, info) in code_score_dict.items() if int(pred) == second_pred])

        ## 第1类别不足3只，则用第2类别补充？暂时仅记录日志后期做分析
        analy_dict = {"max_pred": max_pred, "max_pred_num": max_pred_num, "second_pred": second_pred,
                      "second_pred_num": second_pred_num}
        print(analy_dict)
        rma_logger.info("analy:" + json.dumps(analy_dict, ensure_ascii=False))
        cnum = 0
        for code, (pred, pred_score) in sorted_codes:
            log_dict[code] = {
                "pred": pred,
                "pred_score": pred_score
            }
            if pred == max_pred:
                cnum += 1
                if cnum > 10:
                    continue
                result_dict[code] = {
                "pred": pred,
                "pred_score": pred_score
            }
        # 记录日志
        rma_logger.info("sorted:" + json.dumps(log_dict, ensure_ascii=False))

        ## 上午不再买入，尽量减少在场内时间
        cur_time = datetime.datetime.now().time()
        if self.is_test != "1" and (cur_time < datetime.time(14, 49) or cur_time > datetime.time(14, 57)):
            return json.dumps({"msg": f" 14:57 < cur_time={cur_time} < 14:45, no permit buy!"})

        #### 4. 执行买入操作
        # 买入数量
        target_ind = 0

        for rank, (code, idict) in enumerate(result_dict.items(), 1):
            pred = idict.get('pred', 0)
            score = idict.get('pred_score', 0.0)
            # print(rank, code, pred, score)
            stock = code
            if pred != max_pred:
                continue
            if pred < 11:
                continue
            # print(stock)
            # 已经买入的数量是否达到所需补充的数量
            if target_ind == target_num:
                break

            latest_price = 0.0
            open_price = 0.0
            max_price = 0.0
            # 从实时数据获取最新价格
            try:
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

                #执行买入操作
                ret = dict()
                ret['code'] = stock
                ret['price'] = latest_price
                ret['action'] = 'buy'
                ret['volume'] = buy_volume
                ret['buy_id'] = pred
                ret['buy_score'] = score
                stock_name = self.stock_dict.get(stock.split(".")[0], {}).get("name", "未知")
                ret['name'] = stock_name
                ret['order_id'] = order_id
                if "1" != self.is_test:
                    order_id = xt_trader.order_stock(acc, stock, xtconstant.STOCK_BUY, buy_volume,
                                                     xtconstant.FIX_PRICE, latest_price, strategy_name="mlp_infer_v2")
                    ret['order_id'] = order_id
                    target_ind += 1
                    # 只有线上才更新持仓
                    self.record_trade(
                        action='buy',
                        stock_code=stock,
                        stock_name=stock_name,
                        price=latest_price,
                        volume=buy_volume,
                        order_id=order_id,
                        reason=f"B{pred}",
                        account=account_name
                    )
                self.wt_dict[stock] = 1
                rma_logger.info("b:" + json.dumps(ret, ensure_ascii=False))

                if "1" == self.is_test:
                    print("b:" + json.dumps(ret, ensure_ascii=False))
                # return json.dumps(ret, ensure_ascii=False)

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
    a = MlpInferV2(config="../conf/v1.ini")