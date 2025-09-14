# -*- coding: utf-8 -*-
import sys
import datetime

sys.path.append('../')
import json
import time
from utils import utils
from collections import OrderedDict
##schema:名称,最新价,涨跌幅,涨跌额,成交量,成交额,振幅,最高,最低,今开,昨收,量比,换手率,市盈率-动态,市净率,总市值,流通市值,涨速,5分钟涨跌,60日涨跌幅,年初至今涨跌幅
##schema：['date', 'open', 'close', 'high', 'low', 'volume', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']


class MA_Cross_Checker():
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

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
        ## 查询股票市值
        market_dict = utils.get_stock_info(stock_code)
        ## 总市值和流通值和行业
        # total_mv = market_dict.get("total_mv", -1)
        # circ_mv = market_dict.get("circ_mv", -1)
        # industry = market_dict.get("industry", "")
        # industry_id = market_dict.get("industry_id", "")
        # pe_ttm = market_dict.get("pe_ttm", -1)
        # pb = market_dict.get("pb", -1)
        # print(stock_code, total_mv, industry, industry, pe_ttm, pb)

        ## 此函数只有最新一天的数据
        # df = utils.get_stock_hist_data_em_from_qmt(
        #     stock=stock_code,
        #     start_date='20240501',
        #     end_date='20240507',
        #     # start_date=self.start_date,
        #     # end_date=self.end_date,
        #     data_type='1'
        # )

        ## 此函数只有最新一天的数据
        df = utils.get_stock_hist_minutes_data_em_with_retry(
            stock_code,
                start_date='20240501',
                end_date='20240507',
                # start_date=self.start_date,
                # end_date=self.end_date,
                # data_type='1'
        )
        # print(df['close'].tail(10))
        print(df.columns)
        print(df)

        # 元数据压缩存储
        meta = [
            market_dict.get("total_mv", -1),
            market_dict.get("circ_mv", -1),
            market_dict.get("industry", ""),
            market_dict.get("industry_id", ""),
            market_dict.get("pe_ttm", -1),
            market_dict.get("pb", -1)
        ]

        # 分钟数据压缩存储（数组模式）
        minute_data = []
        for _, row in df.iterrows():
            minute_data.append([
                str(row["date"]),  # 时间戳转字符串
                round(row["open"], 4),  # 保留两位小数
                round(row["close"], 4),
                round(row["high"], 4),
                round(row["low"], 4),
                round(row["振幅"], 4),
                round(row["涨跌幅"], 4),
                round(row["涨跌额"], 4),
                round(row["换手率"], 4)
            ])

        # 构建完整数据结构
        stock_data = OrderedDict()
        stock_data[stock_code] = {
            "i": meta,
            "d": minute_data
        }
        print(stock_data)
        return stock_data


if __name__ == '__main__':
    start  = time.time()
    # 读取stock_code
    stfile = 'D:/tool/dataset/stock_pools.json'
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
    start_date = "20240506"
    end_date = "20240507"
    # formatted_datetime = target_date
    # formatted_datetime = datetime.datetime.now().strftime("%Y%m%d")
    checker = MA_Cross_Checker(start_date, end_date)

    out_file = '../logs/quant_data/hs_quant_base_' + end_date
    # stc_json = open(out_file + '.json', 'w')
    stc_dict = dict()
    target_num = 0
    target_1 = 0
    target_2 = 0
    total_num = len(sc_dict.keys())
    ind = 0
    for code, info_dict in sc_dict.items():
        # if code not in  ["300766", "605398", "300251"]:
        #     continue
        # if not code.startswith("00"):
        #     continue
        ind += 1
        if ind % 100 == 0:
            print(f"processed stock size:{ind}")
        ## 5开头的是etf
        if code.startswith("5"):
            continue

        stock_dict = checker.check_ma_cross(code)
        # print(stock_dict)
        # print(stock_dict.get(code, {}))
        # stc_dict[code] = {'is_cross':is_cross, 'days':days, 'last_close':stock_dict.get(code)}
        # 写入文件（追加模式）
        with open(out_file + ".json", "a", encoding="utf-8") as f:
            json_str = json.dumps(stock_dict, separators=(",", ":"), ensure_ascii=False)
            f.write(json_str + "\n")  # 换行分隔的JSON

        #有交叉的，捞出候选池，线上买点：上升趋势&&无交叉
        # 有交叉的拿出最近多天收盘价，按照时间顺序排序
    # stc_str = json.dumps(stc_dict, ensure_ascii=False, indent=4)
    # stc_json.write(stc_str + '\n')
    # meta = open(out_file+'.meta', 'w')
    # meta_dict = dict()
    # meta_dict["pools_size"] = len(stc_dict)
    # meta_dict["target_1"] = target_1
    # meta_dict["target_2"] = target_2
    # meta_dict["spend_time"] = ( time.time() - start )
    # # row_line = '\t'.join(('pools_size:', str(len(stc_dict)), "; target_num:", str(target_num)))
    # row_line = json.dumps(meta_dict, ensure_ascii=False, indent=4)
    # meta.write(row_line+'\n')
