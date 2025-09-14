# -*- coding: utf-8 -*-
import os
import sys
import glob
import json
import datetime

sys.path.append('../')
import json
import time
from utils import utils
#schema:名称,最新价,涨跌幅,涨跌额,成交量,成交额,振幅,最高,最低,今开,昨收,量比,换手率,市盈率-动态,市净率,总市值,流通市值,涨速,5分钟涨跌,60日涨跌幅,年初至今涨跌幅
"""
强度区间	数值范围	颜色标识
极度弱势	0.00 ≤ score < 0.50	🔴 红色
弱势震荡	0.50 ≤ score < 1.20	🟠 橙色
中等活跃	1.20 ≤ score < 2.00	🟡 黄色
强势行情	2.00 ≤ score < 3.50	🟢 绿色
极端过热	score ≥ 3.50	⚠️ 紫色

极端过热需要空仓

动态调整机制
修正因子	对区间的影响
炸板率 >50%	当前区间降级一档（如1.45→弱势震荡）
连板高度 ≥7	当前区间升级一档（如1.45→强势行情）
总涨停数 <30	当前区间降级一档

识别：
假强势（高score+高炸板率）
真启动（低score+炸板率快速下降）
"""


class DailyLimitStrength():
    def __init__(self, target_date):
        self.target_date = target_date
        # pools_file = './logs/dragon_v5_data.20241130'
        pools_file = self.get_latest_file('../logs', 'dragon_v5_data')
        # pools_file = '../logs\dragon_v5_data_20250417.json'
        print('[DEBUG]load_file:', pools_file)

        self.stock_dict = json.load(open(pools_file))
        print('[INIT]load target size:', len(self.stock_dict))
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


    def do(self):
        """
        均线交叉检测函数（支持多日连续检测）
        :param stock_code: 6位股票代码 如：'600519'
        :param days: 需要检测的天数范围（默认检测最近2天）
        :return: bool 是否在指定天数内出现交叉
        """
        # start_date='20210101',  # 起始日期设为足够早
        continue_limit_up_strength_dict = dict()
        ## 连板数
        continue_limit_up_num = 0
        ## 炸板数
        zha_limit_up_num = 0
        ## 总涨停数量
        all_limit_up_num = 0
        continue_limit_up_list = list()
        ## 总涨停数量
        for code, info_dict in self.stock_dict.items():
            name = info_dict.get("name", "")
            ## 连续涨停个股数量
            continuous_up_limit_days = info_dict.get("continuous_up_limit_days", -1)
            # 当天最高价触板
            is_today_high_limit_up = info_dict.get("is_today_high_limit_up", -1)
            # 当天收盘是否涨停
            is_today_limit_up = info_dict.get("is_today_limit_up", -1)

            if continuous_up_limit_days == -1 or is_today_high_limit_up == -1 or is_today_limit_up == -1:
                continue
            # print(f"[DEBUG]{code}=={str(continuous_up_limit_days)}=={continuous_up_limit_days}")
            if continuous_up_limit_days >= 2:
                continue_limit_up_num += 1
                continue_limit_up_list.append(continuous_up_limit_days)
            if is_today_high_limit_up == 1 and is_today_limit_up <= 0:
                zha_limit_up_num += 1
            if is_today_limit_up:
                all_limit_up_num += 1
            # name = info_dict.get("name", "")
            # name = info_dict.get("name", "")

        continue_limit_up_strength_dict["continue_limit_up_num"] = continue_limit_up_num
        continue_limit_up_strength_dict["zha_limit_up_num"] = zha_limit_up_num
        continue_limit_up_strength_dict["all_limit_up_num"] = all_limit_up_num
        continue_limit_up_strength_dict["continue_limit_up_list"] = continue_limit_up_list
        score, zha_rate, avg_continue_limit_up = self.calculate_board_strength(continue_limit_up_strength_dict)
        continue_limit_up_strength_dict["score"] = score
        continue_limit_up_strength_dict["zha_rate"] = zha_rate
        continue_limit_up_strength_dict["avg_continue_limit_up"] = avg_continue_limit_up
        #print(rate)
        return continue_limit_up_strength_dict

    def calculate_board_strength(self, board_data):
        """
        计算连板强度

        :param board_data: dict, 连板及市场情绪数据，例如
                           {
                               "连板数": 15,
                               "总涨停数": 40,
                               "连板高度列表": [2, 3, 2, 4, 5, 1],  # 每个连板股票的高度
                               "炸板数": 10
                           }
        :return: float, 连板强度
        """
        # 连板数
        continue_limit_up_num = board_data['continue_limit_up_num']
        # 总涨停数
        all_limit_up_num = board_data['all_limit_up_num']
        # 炸板数
        zha_limit_up_num = board_data['zha_limit_up_num']
        # 连板高度列表
        continue_limit_up_list = board_data['continue_limit_up_list']

        if all_limit_up_num == 0:  # 避免除以零
            return 0, 0, 0

        # 计算平均连板高度
        avg_continue_limit_up = 0
        if continue_limit_up_num > 0:
            avg_continue_limit_up = sum(continue_limit_up_list) / len(continue_limit_up_list)
        else:
            avg_continue_limit_up = 0

        # 计算炸板率
        zha_rate = zha_limit_up_num / (zha_limit_up_num + continue_limit_up_num)  # 炸板尝试总数 = 炸板数 + 连板数

        # 综合公式
        # 连板强度 = (连板数 * 平均连板高度) / (总涨停数 * 炸板率 + 1)
        strength = (continue_limit_up_num  * avg_continue_limit_up) / (all_limit_up_num*zha_rate + 1.0)
        return round(strength, 4), zha_rate, avg_continue_limit_up

if __name__ == '__main__':
    start  = time.time()

    # target_date = "20500101"
    # target_date = "20250418"
    # formatted_datetime = target_date
    formatted_datetime = datetime.datetime.now().strftime("%Y%m%d")
    checker = DailyLimitStrength(formatted_datetime)

    out_file = '../logs/v1_daily_limit_strength_data_' + formatted_datetime
    stc_json = open(out_file + '.json', 'w')
    stc_dict = checker.do()
    stc_str = json.dumps(stc_dict, ensure_ascii=False, indent=4)
    stc_json.write(stc_str + '\n')
    meta = open(out_file+'.meta', 'w')
    meta_dict = dict()
    meta_dict["pools_size"] = len(stc_dict)
    meta_dict["spend_time"] = ( time.time() - start )
    # row_line = '\t'.join(('pools_size:', str(len(stc_dict)), "; target_num:", str(target_num)))
    row_line = json.dumps(meta_dict, ensure_ascii=False, indent=4)
    meta.write(row_line+'\n')
