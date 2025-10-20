# -*- coding: utf-8 -*-
"""
合并两个 Excel
规则：
1. 左表 key = code.split('.')[0]
2. 右表 key = 股票代码
3. 合并方式：left（以左表为主）
输出：merge_result.xlsx（同目录）
"""
import pandas as pd
import os

# 1. 读入两个文件（改成自己的路径）
left_file  = r'D:/tool/dataset/temp/rtrma/analy_20251014.xlsx'   # 含 code/name/.../diff
right_file = r'D:\tool\dataset\backtest_summary_rmav4.2_baseline/个股表现_20251004_011839.xlsx'        # 含 股票代码/收益率/...
out_file = os.path.join(os.path.dirname(left_file), 'merge_result_20251014.xlsx')

left  = pd.read_excel(left_file)
right = pd.read_excel(right_file)

# 2. 生成 key
left['key']  = left['code'].astype(str).str.split('.').str[0]
right['key'] = right['股票代码'].astype(str).str.zfill(6)   # 防止 000001 被读成 1

# 3. 合并
merge_df = left.merge(right.drop(columns='股票代码'),   # 去掉重复的“股票代码”列
                      on='key',
                      how='left')

# 4. 清理辅助列并保存
merge_df.drop(columns='key', inplace=True)
merge_df.to_excel(out_file, index=False)
print(f'合并完成 -> {out_file}')