# coding=utf8
# -*- coding: utf-8 -*-
import sys
import datetime
import numpy as np

sys.path.append('../')
from utils import utils  # 假设存在获取指数数据的工具函数


class MarketStrengthAnalyzer:
    def __init__(self, target_date):
        self.target_date = target_date
        self.index_list = [
            ('000001.SH', '上证指数'),
            ('399001.SZ', '深证成指'),
            ('000016.SH', '上证50'),
            ('000905.SH', '中证500'),
            ('399006.SZ', '创业板指')
        ]

    def get_index_data(self, index_code):
        """获取指数历史数据"""
        return utils.get_stock_hist_data_em_with_retry(
            index_code,
            start_date='20210101',
            end_date=self.target_date,
            data_type='D'
        )

    def calculate_market_strength(self):
        """
        计算大盘环境强度（综合主要指数）
        返回：强度评分（0-100）、趋势方向（1上升/-1下降）
        """
        strength_score = 0
        trend_direction = 0

        # 1. 计算各指数均线状态
        ma_status = {}
        for code, name in self.index_list:
            df = self.get_index_data(code)
            if len(df) < 20: continue

            # 计算均线
            df['MA5'] = df['close'].rolling(5).mean()
            df['MA20'] = df['close'].rolling(20).mean()

            # 当前价格与均线关系
            last_close = df['close'].iloc[-1]
            ma5 = df['MA5'].iloc[-1]
            ma20 = df['MA20'].iloc[-1]

            # 均线多头排列得分
            ma_score = 0
            if last_close > ma5 > ma20:
                ma_score = 2
            elif ma5 > ma20:
                ma_score = 1

            # 量能变化得分
            volume_score = 1 if df['volume'].iloc[-1] > df['volume'].iloc[-5:-1].mean() else 0

            ma_status[code] = {
                'ma_score': ma_score,
                'volume_score': volume_score,
                'trend': 1 if ma5 > df['MA5'].iloc[-5] else -1
            }

        # 2. 综合评分计算
        weight_map = {
            '000001.SH': 0.4,  # 上证指数权重40%
            '399001.SZ': 0.3,  # 深成指30%
            '000016.SH': 0.15,  # 上证50
            '399006.SZ': 0.15  # 创业板
        }

        total_score = 0
        for code, w in weight_map.items():
            if code not in ma_status: continue
            total_score += (ma_status[code]['ma_score'] + ma_status[code]['volume_score']) * w * 20

        # 3. 趋势方向判断
        up_count = sum(1 for s in ma_status.values() if s['trend'] > 0)
        trend_direction = 1 if up_count >= 3 else -1

        # 将得分限制在0-100区间
        strength_score = np.clip(total_score, 0, 100)
        return strength_score, trend_direction

    def calculate_index_strength(self, days=5):
        """
        计算指数间相对强度
        返回：{
            'strongest_index': 最强指数代码,
            'weakest_index': 最弱指数代码,
            'relative_strength': 各指数强度值
        }
        """
        strength_data = {}

        for code, name in self.index_list:
            df = self.get_index_data(code)
            if len(df) < days: continue

            # 计算N日收益率
            returns = (df['close'].iloc[-1] / df['close'].iloc[-days] - 1) * 100

            # 计算相对强弱指标（RS）
            avg_vol = df['volume'].iloc[-days:].mean()
            prev_avg_vol = df['volume'].iloc[-2 * days:-days].mean()
            vol_ratio = avg_vol / prev_avg_vol if prev_avg_vol > 0 else 1

            # 强度公式：收益率 * 量能变化系数
            strength = returns * np.log1p(vol_ratio)
            strength_data[code] = round(strength, 2)

        # 排序找出最强和最弱
        sorted_strength = sorted(strength_data.items(), key=lambda x: x[1], reverse=True)
        return {
            'strongest_index': sorted_strength[0][0] if sorted_strength else None,
            'weakest_index': sorted_strength[-1][0] if sorted_strength else None,
            'relative_strength': strength_data
        }


# 集成到原有策略中
class MA_Cross_Checker(MarketStrengthAnalyzer):
    def __init__(self, target_date):
        super().__init__(target_date)
        self.market_strength_threshold = 60  # 强度合格阈值

    def check_market_condition(self):
        """综合判断市场环境"""
        strength_score, trend = self.calculate_market_strength()
        index_strength = self.calculate_index_strength()

        # 环境判断逻辑
        market_ok = strength_score >= self.market_strength_threshold
        index_ok = index_strength['strongest_index'] in ['000001.SH', '399001.SZ']

        return {
            'market_ok': market_ok and index_ok,
            'strength_score': strength_score,
            'main_trend': trend,
            'index_strength': index_strength
        }


# 使用示例
if __name__ == '__main__':
    analyzer = MarketStrengthAnalyzer("20231020")

    # 获取大盘强度
    strength, trend = analyzer.calculate_market_strength()
    print(f"大盘综合强度：{strength} 趋势方向：{'上涨' if trend == 1 else '下跌'}")

    # 获取指数相对强度
    index_strength = analyzer.calculate_index_strength()
    print("指数强度排名：")
    for code, strength in index_strength['relative_strength'].items():
        print(f"{code}: {strength}")