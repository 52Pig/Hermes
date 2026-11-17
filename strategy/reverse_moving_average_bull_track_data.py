# -*- coding: utf-8 -*-
import sys
import datetime

sys.path.append('../')
import json
import time
from utils import utils
##schema:名称,最新价,涨跌幅,涨跌额,成交量,成交额,振幅,最高,最低,今开,昨收,量比,换手率,市盈率-动态,市净率,总市值,流通市值,涨速,5分钟涨跌,60日涨跌幅,年初至今涨跌幅



class MA_Cross_Checker():
    def __init__(self, target_date):
        self.target_date = target_date

    def check_ma_cross(self, stock_code, days=2):
        """
        均线交叉检测函数（支持多日连续检测）
        :param stock_code: 6位股票代码 如：'600519'
        :param days: 需要检测的天数范围（默认检测最近2天）
        :return: bool 是否在指定天数内出现交叉
        """
        # start_date='20210101',  # 起始日期设为足够早
        stock_dict = dict()
        stock_dict[stock_code] = {}
        try:
            ## 查询股票市值
            market_dict = utils.get_stock_info(stock_code)
            ## 总市值和流通值和行业
            total_mv = market_dict.get("total_mv", -1)
            circ_mv = market_dict.get("circ_mv", -1)
            industry = market_dict.get("industry", "")
            industry_id = market_dict.get("industry_id", "")
            pe_ttm = market_dict.get("pe_ttm", -1)
            pb = market_dict.get("pb", -1)
            # print(stock_code, total_mv, industry, industry, pe_ttm, pb)

            # 获取历史数据（获取足够长的数据确保计算）
            df = utils.get_stock_hist_data_em_with_retry(
                stock_code,
                '20240601',
                # '20500101',
                # '20250429'
                self.target_date,
                'D'
            )
            # print(df['close'].tail(10))
            # print(df.columns)

            # 数据校验
            if len(df) < 40:
                print(f"{stock_code} 数据不足，仅获取到{len(df)}个交易日数据")
                return False, stock_dict

            ## 股性检测，最近半年有过涨停
            is_up_limit_half_year = False
            half_year_days = 254
            last_days_close = df['close'].tail(half_year_days)
            ## 0. 股性检测，最近半年内是否有过涨停
            for i in range(1, min(half_year_days, len(last_days_close))):
                prev_price = last_days_close.iloc[-i - 1]
                cur_price = last_days_close.iloc[-i]

                # 精确涨停判断（考虑四舍五入）
                # 规则：当前收盘价 >= 前一日收盘价 * 1.099（保留两位小数后等于10%涨幅）
                if abs(cur_price - round(prev_price * 1.1, 2)) < 1e-6:  # 浮点数精度处理
                    is_up_limit_half_year = True
                    break
            ## 1. 倒序检查 连续涨停天数, 最近5天不能有涨停
            is_not_up_limit = False
            up_limit_days = 0
            ## 统计最近20天股价涨停的天数
            max_check = min(5, len(last_days_close) - 1)  # 需要至少两天数据（前一天+当天）
            for i in range(1, max_check + 1):
                prev_price = last_days_close.iloc[-i - 1]  # 前一日收盘价
                cur_price = last_days_close.iloc[-i]  # 当前日收盘价

                # 精确涨停判断（四舍五入到分）
                expected_price = round(prev_price * 1.1, 2)
                # print(f"expected_price={expected_price};cur_price={cur_price};prev_price={prev_price}")
                if abs(cur_price - expected_price) < 1e-6:  # 处理浮点精度
                    up_limit_days += 1
                    # print(f"up_limit_days={up_limit_days}")

            if up_limit_days == 0:
                is_not_up_limit = True


            ## 2.近期涨幅不能过大
            latest_price = df['close'].iloc[-1]
            last_7_close = df['close'].tail(7)
            diff = (last_7_close.max() - last_7_close.min()) / last_7_close.min()
            is_last_1d_zf = (latest_price - df['close'].iloc[-2]) / df['close'].iloc[-2]
            is_last_2d_zf = (df['close'].iloc[-2] - df['close'].iloc[-3] ) / df['close'].iloc[-3]

            is_last_3d_zf = (df['close'].iloc[-3] - df['close'].iloc[-4]) / df['close'].iloc[-4]
            is_last_4d_zf = (df['close'].iloc[-4] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            is_last_5d_zf = (df['close'].iloc[-5] - df['close'].iloc[-6]) / df['close'].iloc[-6]

            stock_dict[stock_code]["is_last_1d_zt"] = is_last_1d_zf
            stock_dict[stock_code]["is_last_2d_zt"] = is_last_2d_zf
            stock_dict[stock_code]["last_close"] = ','.join([str(ele) for ele in df['close'].dropna().to_list()])

            # 4.均线交叉
            ma_windows = [5, 10, 20, 30]
            for n in ma_windows:
                df[f'MA{n}'] = df['close'].rolling(window=n).mean()
                # print(df[f'MA{n}'].tail(35))
            # 去除空值
            valid_df = df.dropna(subset=['MA30']).reset_index(drop=True)
            # 记录信息
            # stock_dict[stock_code]["MA3"] = valid_df['MA3'].iloc[-1]
            stock_dict[stock_code]["MA5"] = valid_df['MA5'].iloc[-1]
            stock_dict[stock_code]["MA10"] = valid_df['MA10'].iloc[-1]
            stock_dict[stock_code]["MA20"] = valid_df['MA20'].iloc[-1]
            stock_dict[stock_code]["MA30"] = valid_df['MA30'].iloc[-1]

            # 提取指定天数内的数据
            recent_data = valid_df.iloc[-7:].copy()
            # print(df)

            # 判断当天 MA5 是否大于过去 MA5
            # 5.均线上升
            is_up_ma = False
            # 6. 今日 MA5 最低 MA10>MA20>MA30>MA5
            is_ma5_lowest = False
            # 7. 今日 MA5>MA10>MA20>MA30
            is_last_1d_ma5_not_lowest = False
            if len(valid_df) > 1:  # 需要至少两天数据进行对比
                is_up_ma = (
                        valid_df['MA5'].iloc[-1] > valid_df['MA5'].iloc[-2] )
                        # valid_df['MA10'].iloc[-1] > valid_df['MA10'].iloc[-2] and
                        # valid_df['MA20'].iloc[-1] >= valid_df['MA20'].iloc[-2] and
                        # valid_df['MA30'].iloc[-1] >= valid_df['MA30'].iloc[-2])
                is_ma5_lowest = (valid_df["MA10"].iloc[-1] >= valid_df["MA5"].iloc[-1] and
                                         valid_df["MA20"].iloc[-1] >= valid_df["MA5"].iloc[-1] and
                                         valid_df["MA30"].iloc[-1] >= valid_df["MA5"].iloc[-1])
                is_last_1d_ma5_not_lowest = valid_df["MA5"].iloc[-1] >= valid_df["MA10"].iloc[-1] or \
                                            valid_df["MA5"].iloc[-1] >= valid_df["MA20"].iloc[-1] or \
                                            valid_df["MA5"].iloc[-1] >= valid_df["MA30"].iloc[-1]
            # print(f"is_up_ma:{is_up_ma}; is_not_up_limit:{is_not_up_limit};is_up_limit_half_year={is_up_limit_half_year}")
            # print(f"is_ma5_lowest:{is_ma5_lowest}")
            # print(f'is_last_1d_ma5_not_lowest:{is_last_1d_ma5_not_lowest};{valid_df["MA5"].iloc[-1]};{valid_df["MA10"].iloc[-1]}; {valid_df["MA20"].iloc[-1]}, {valid_df["MA30"].iloc[-1]}')
            # # 7. MA3 与其他 MA 交叉
            # cross_found = False
            # ma_columns = [f'MA{n}' for n in ma_windows]
            # other_mas = [ma for ma in ma_columns if ma not in  ["MA3", "MA5"]]
            # for other_ma in other_mas:
            #     recent_data["diff"] = recent_data["MA5"] - recent_data[other_ma]
            #     sign_changes = (recent_data['diff'] * recent_data['diff'].shift(1)) <= 0
            #     sign_changes = sign_changes.iloc[1:]  # 排除首日NaN
            #     if sign_changes.any():
            #         cross_found = True
            #         break

            # # 8.检查斜率条件：MA3
            # # 计算MA3的斜率
            # is_max_slope_ma3 = True
            # slope_ma3 = recent_data["MA3"].iloc[-1] - recent_data["MA3"].iloc[-2]
            # if slope_ma3 > 0:
            #     # 检查其他MA的斜率
            #     for ma in other_mas:
            #         slope = recent_data[ma].iloc[-1] - recent_data[ma].iloc[-2]
            #         if slope >= slope_ma3:
            #             is_max_slope_ma3 = False
            #             break
            # else:
            #     is_max_slope_ma3 = False


            # for (ma1, ma2) in combinations(ma_columns, 2):
            #     # 计算每日差值符号变化
            #     recent_data['diff'] = recent_data[ma1] - recent_data[ma2]
            #     sign_changes = (recent_data['diff'] * recent_data['diff'].shift(1)) <= 0
            #
            #     # 排除首日的NaN
            #     sign_changes = sign_changes.iloc[1:]
            #
            #     if sign_changes.any():
            #         cross_found = True
            #         break
            # stock_dict["cross_found"] = cross_found
            stock_dict[stock_code]["is_up_ma"] = is_up_ma
            # stock_dict["is_max_slope_ma3"] = is_max_slope_ma3
            stock_dict[stock_code]["is_up_limit_half_year"] = is_up_limit_half_year
            stock_dict[stock_code]["up_limit_days"] = up_limit_days
            stock_dict[stock_code]["total_mv"] = total_mv
            stock_dict[stock_code]["circ_mv"] = circ_mv
            stock_dict[stock_code]["industry"] = industry
            stock_dict[stock_code]["industry_id"] = industry_id
            stock_dict[stock_code]["pe_ttm"] = pe_ttm
            stock_dict[stock_code]["pb"] = pb
            # print(stock_dict)

            # if cross_found:
            #     print('cross_found:', stock_dict)
            # if ma_up:
            #     print('ma_up:', stock_dict)
            # if is_max_slope_ma3:
            #     print('is_max_slope_ma3:', stock_dict)
            # last_days_close = df[''].tail(half_year_days)

            ##schema:名称,最新价,涨跌幅,涨跌额,成交量,成交额,振幅,最高,最低,今开,昨收,量比,换手率,市盈率-动态,市净率,总市值,流通市值,涨速,5分钟涨跌,60日涨跌幅,年初至今涨跌幅

            # print(stock_code, valid_df['总市值'].iloc[-1])
            # print(stock_code, valid_df.columns)
            # valid_df["MA10"].iloc[-1]
            stock_dict[stock_code]["last_5d_diff"] = diff
            stock_dict[stock_code]["is_ma5_lowest"] = 1 if is_ma5_lowest else 0
            stock_dict[stock_code]["is_not_up_limit"] = 1 if is_not_up_limit else 0
            stock_dict[stock_code]["is_up_limit_half_year"] = 1 if is_up_limit_half_year else 0
            # if total_mv > 320:
            #     ## 市值太高不好拉升
            #     return False, stock_dict
            # el
            # if latest_price < 2.0 or latest_price > 20.0:
            if latest_price < 2.0:
                ## 单价太高不易拉升
                # print("[DEBUG]filter latest price", stock_code, stock_name)
                return False, stock_dict
            # elif diff >= 0.08:
                # 最近涨幅不能过大，判断差值是否小于0.08
                # print(f"{stock_code} {diff} {last_7_close.min()} {last_7_close.max()} 最近7天的收盘价最大最小差值大于0.08")
                # return False, stock_dict
            elif is_last_1d_zf > 0.06 or is_last_2d_zf > 0.06 or is_last_3d_zf > 0.096 or is_last_3d_zf > 0.096 or is_last_3d_zf > 0.096:
                return False, stock_dict
            elif is_ma5_lowest and is_not_up_limit and is_up_limit_half_year:
                return True, stock_dict
            else:
                return False, stock_dict

                # return True if is_up_ma and is_last_1d_ma5_lowest and is_not_up_limit and is_up_limit_half_year else False, stock_dict
                # return True if is_up_ma and is_last_1d_ma5_lowest and is_last_1d_ma5_not_lowest and is_up_limit_half_year else False, stock_dict
        except Exception as e:
            print(f"策略执行异常：{stock_code}  {str(e)}")
            return False, stock_dict


if __name__ == '__main__':
    start  = time.time()
    # 读取stock_code
    stfile = '../dataset/stock_pools.json'
    sc_dict = json.load(open(stfile))
    # print(sc_json.get("600259.SH"))
    # st_list = [(ele.split('.')[0], 2) for ele in sc_dict.keys()]
    # quit()

    # 测试
    #test_cases = [
        # ('600519', 2),  # 贵州茅台
        # ('000001', 2),  # 平安银行
        # ('300750', 2),  # 宁德时代
        # ('000560', 2),  # 我爱我家
        # ('300766', 10),  # 每日互动
        #('000532', 5),  # 华金资本
        # ('600530', 5),  # 交大昂立
    #    ('603759', 5)  # 海天股份
    #]
    # target_date = "20500101"
    # formatted_datetime = target_date
    # target_date = "20250416"
    # formatted_datetime = target_date
    formatted_datetime = datetime.datetime.now().strftime("%Y%m%d")
    checker = MA_Cross_Checker(formatted_datetime)

    out_file = '../logs/reverse_moving_average_bull_track_' + formatted_datetime
    stc_json = open(out_file + '.json', 'w')
    stc_dict = dict()
    target_num = 0
    total_num = len(sc_dict.keys())
    ind = 0
    for code, info_dict in sc_dict.items():
        # if code != "000560":
        #     continue
        ind += 1
        if ind % 100 == 0:
            print(f"processed stock size:{ind}")
        ## 5开头的是etf
        if code.startswith("5"):
            continue

        is_cross, stock_dict = checker.check_ma_cross(code)
        # print(stock_dict)
        # print(stock_dict.get(code, {}))
        # stc_dict[code] = {'is_cross':is_cross, 'days':days, 'last_close':stock_dict.get(code)}

        #print(f"股票{code}最近{days}个交易日内均线交叉状态：{is_cross}")
        stc_dict[code] = sc_dict.get(code)
        ## 是否当天标的
        if is_cross:
            stc_dict[code]["is_target"] = '1'
            target_num += 1
        else:
            stc_dict[code]["is_target"] = '0'
        ## 无论是否当天标的，都需要历史数据辅助做决策
        stc_dict[code]["last_close"] = stock_dict.get(code, {}).get("last_close", "")
        stc_dict[code]["MA3"] = stock_dict.get(code, {}).get("MA3", 1000)
        stc_dict[code]["MA5"] = stock_dict.get(code, {}).get("MA5", 1000)
        stc_dict[code]["MA10"] = stock_dict.get(code, {}).get("MA10", 1000)
        stc_dict[code]["MA20"] = stock_dict.get(code, {}).get("MA20", 1000)
        stc_dict[code]["MA30"] = stock_dict.get(code, {}).get("MA30", 1000)

        # print(stock_dict.get(code, {}))
        stc_dict[code]["up_limit_days"] = stock_dict.get(code, {}).get("up_limit_days", -1)

        stc_dict[code]["total_mv"] = stock_dict.get(code, {}).get("total_mv", -1)
        stc_dict[code]["circ_mv"] = stock_dict.get(code, {}).get("circ_mv", -1)
        stc_dict[code]["industry"] = stock_dict.get(code, {}).get("industry", -1)
        stc_dict[code]["industry_id"] = stock_dict.get(code, {}).get("industry_id", -1)
        stc_dict[code]["pe_ttm"] = stock_dict.get(code, {}).get("pe_ttm", -1)
        stc_dict[code]["pb"] = stock_dict.get(code, {}).get("pb", -1)
        stc_dict[code]["last_5d_diff"] = stock_dict.get(code, {}).get("last_5d_diff", -1)

        stc_dict[code]["is_ma5_lowest"] = stock_dict.get(code, {}).get("is_ma5_lowest", -1)
        stc_dict[code]["is_not_up_limit"] = stock_dict.get(code, {}).get("is_not_up_limit", -1)
        stc_dict[code]["is_up_limit_half_year"] = stock_dict.get(code, {}).get("is_up_limit_half_year", -1)

        stc_dict[code]["last_1d_zf"] = stock_dict.get(code, {}).get("is_last_1d_zf", -1)
        stc_dict[code]["last_2d_zf"] = stock_dict.get(code, {}).get("is_last_2d_zf", -1)

        #有交叉的，捞出候选池，线上买点：上升趋势&&无交叉
        # 有交叉的拿出最近多天收盘价，按照时间顺序排序
    stc_str = json.dumps(stc_dict, ensure_ascii=False, indent=4)
    stc_json.write(stc_str + '\n')
    meta = open(out_file+'.meta', 'w')
    meta_dict = dict()
    meta_dict["pools_size"] = len(stc_dict)
    meta_dict["target_num"] = target_num
    meta_dict["spend_time"] = ( time.time() - start )
    # row_line = '\t'.join(('pools_size:', str(len(stc_dict)), "; target_num:", str(target_num)))
    row_line = json.dumps(meta_dict, ensure_ascii=False, indent=4)
    meta.write(row_line+'\n')
