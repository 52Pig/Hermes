# -*- coding: utf-8 -*-
import os
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

    def do(self, stock_code, df, formatted_datetime):
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
        market_dict = dict()
        # market_dict = utils.get_stock_info(stock_code)
        ## 总市值和流通值和行业
        # total_mv = market_dict.get("total_mv", -1)
        # circ_mv = market_dict.get("circ_mv", -1)
        # industry = market_dict.get("industry", "")
        # industry_id = market_dict.get("industry_id", "")
        # pe_ttm = market_dict.get("pe_ttm", -1)
        # pb = market_dict.get("pb", -1)
        # print(stock_code, total_mv, industry, industry, pe_ttm, pb)

        # 获取历史数据
        ##df schema：['date', 'open', 'close', 'high', 'low', 'volume', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
        if df.get(stock_code) is None:
            print(f"[WARN]df of stock={stock_code} is None")
            return
        # print(df['close'].tail(10))
        # print(df.columns)
        # print(df)
        # quit()
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
        for _, row in df[stock_code].iterrows():
            if not row['date'].startswith(formatted_datetime):
                continue
            minute_data.append([
                str(row["date"]),  # 时间戳转字符串
                round(row["open"], 4),  # 保留两位小数
                round(row["close"], 4),
                round(row["high"], 4),
                round(row["low"], 4),
                round(row["volume"]),
                round(row["amount"]),
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
        # print(stock_data)
        return stock_data


if __name__ == '__main__':
    do_start  = time.time()
    config = {
        'stock_pool_file': 'D:/tool/dataset/stock_pools.json',
        'output_dir': 'D:/tool/dataset/infer_data/',
        'output_file_prefix': 'hs_quant_base_'
    }
    os.makedirs(config['output_dir'], exist_ok=True)
    # 起始日期
    # trade_days = utils.get_a_share_trade_days_from_ak(start_date='20250601')
    formatted_datetime = datetime.datetime.now().strftime("%Y%m%d")
    # formatted_datetime = "20250827"
    trade_days = utils.get_a_share_trade_days(start_date=formatted_datetime)
    # trade_days = trade_days[:81]
    # print(trade_days)
    if len(trade_days) == 0:
        print(f"[DEBUG]{formatted_datetime} is not a valid trade day! program quit!")
        quit()
    # 读取stock_code
    # stfile = '../dataset/stock_pools.json'
    stfile = config['stock_pool_file']
    sc_dict = json.load(open(stfile))
    # print(sc_json.get("600259.SH"))
    # st_list = [(ele.split('.')[0], 2) for ele in sc_dict.keys()]
    # quit()
    # target_date = "20500101"
    # formatted_datetime = target_date
    for dt in trade_days:
        # start_date = "20240508"
        # end_date = "20240508"
        # formatted_datetime = target_date
        # formatted_datetime = datetime.datetime.now().strftime("%Y%m%d")
        start_date = dt
        end_date = dt
        checker = MA_Cross_Checker(start_date, end_date)

        # out_file = '../logs/quant_data/hs_quant_base_' + end_date
        # out_file = 'F:/tools/dataset/quant_data/hs_quant_base_' + end_date
        out_file = os.path.join(config['output_dir'], config['output_file_prefix'] + end_date + ".json")
        print(f"out_file={out_file}")
        fw_file = open(out_file , "w", encoding="utf-8")
        # stc_json = open(out_file + '.json', 'w')
        stc_dict = dict()
        target_num = 0
        target_1 = 0
        target_2 = 0
        total_num = len(sc_dict.keys())
        ind = 0


        code_list = list(set([code for code, info_dict in sc_dict.items()]))
        _start = time.time()
        df = utils.get_batch_stock_data_em_from_qmt(stock_list=code_list, start_date='20250825',
                                              end_date='20500101',
                                              data_type='1',
                                              count=-1)
        _end = time.time()
        print(f"get_batch_stock_data spend:{(_end - _start)}s")
        # print(df)


        for code, info_dict in sc_dict.items():
            ind += 1
            # if code != '002970':
            #     continue

            if ind % 100 == 0:
                print(f"processed num={ind} code={code} dt={dt}")
            # if code not in ["688758"]:
            # if code not in ["300766", "605398", "300251"]:
            #     continue
            # if not code.startswith("00"):
            #     continue
            # ind += 1
            # if ind % 100 == 0:
            #     print(f"processed stock size:{ind}")
            ## 5开头的是etf
            if code.startswith("5"):
                continue

            stock_dict = checker.do(code, df, end_date)
            if stock_dict is None:
                continue
            stock = info_dict.get("code", "")
            name = info_dict.get("name", "")
            stock_dict[code]["code"] = stock
            stock_dict[code]["name"] = name
            # print(stock_dict)
            # print(stock_dict.get(code, {}))
            # stc_dict[code] = {'is_cross':is_cross, 'days':days, 'last_close':stock_dict.get(code)}
            # 写入文件（追加模式）
            json_str = json.dumps(stock_dict, separators=(",", ":"), ensure_ascii=False)
            fw_file.write(json_str + "\n")  # 换行分隔的JSON

    do_end = time.time()
    print(f"task finish spend time={do_end - do_start}")