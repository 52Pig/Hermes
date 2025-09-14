# -*- coding: utf-8 -*-
import os, sys
import json
import numpy as np
from xtquant import xtdata


def gen_stock_pools():
    ''' 生成准入池
    :return:
    '''
    index_code = '沪深A股'
    index_stocks = xtdata.get_stock_list_in_sector(index_code)
    # print("[DEBUG]hs=", len(index_stocks), index_stocks)
    ## 筛选出有效准入池
    recall_stock = list()
    stc_dict = dict()
    for stock_code in index_stocks:
        if not stock_code.endswith(".SH") and not stock_code.endswith(".SZ"):
            continue
        ## 排除ST的股票
        instrument_detail = xtdata.get_instrument_detail(stock_code)
        # print('--------------', json.dumps(instrument_detail))
        # if '300615' in stock_code:
        # print('[DEBUG]instrument_detail=',instrument_detail)
        stock_name = ''
        if instrument_detail is None:
            continue
        if instrument_detail is not None:
            stock_name = instrument_detail.get("InstrumentName", "")
            if "ST" in stock_name:
                # print("[DEBUG]filter_ST=", stock_code, stock_name)
                continue
        # 排除创业板
        if stock_code.startswith("3"):
            # print("[DEBUG]filter_chuangyeban=", stock_code, stock_name)
            continue
        ## 是否停牌
        ins_status = instrument_detail.get("InstrumentStatus", 0)
        if ins_status >= 1:
            print("[DEBUG]=instrumentStatus", ins_status, stock_code, stock_name)
            continue
        ## 查询流通量
        floatVol = instrument_detail.get("FloatVolume", 0)
        ## 查询总量
        totalVol = instrument_detail.get("TotalVolume", 0)
        code = stock_code.split('.')[0].strip()
        stc_dict[code] = {"code": stock_code, "name": stock_name, "FloatVolume": floatVol, "TotalVolume": totalVol}

    print('pools_size:', len(stc_dict))

    if len(stc_dict) > 10:
        data_path = "D:/tool/dataset"
        stc_json = open(os.path.join(data_path, 'stock_pools.json'), 'w')
        stc_str = json.dumps(stc_dict, ensure_ascii=False, indent=4)
        stc_json.write(stc_str+'\n')

        meta = open(os.path.join(data_path, 'stock_pools.meta'), 'w')
        row_lines = '\t'.join(('pools_size', str(len(stc_dict))))
        meta.write(row_lines + '\n')


if __name__ == '__main__':
    gen_stock_pools()