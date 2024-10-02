#coding=utf8

import os
import json


ppath = 'clean_sample'

## 加载索引
d = json.load(open('lookup_stock_code.json'))
# dt，pprice，high，low，close, volume, stock_code_hash, year_day_str, year_month_str, month_day_str, hour_of_day_str, minute_of_hour_str
t = dict()
for k, v in d.items():
    t[v] = k

filename = os.listdir(ppath)
stock_name = '000560.SZ'
fw_file = open(stock_name, 'w')

for fname in filename:
    for i, line in enumerate(open(os.path.join(ppath, fname))):
        line = line.rstrip('\r\n')
        lines = line.split('\t')
        if len(lines) < 7:
            continue
        tm = lines[0].strip()
        sth = lines[6].strip()
        stock_code = t[sth]
        if stock_code != stock_name:
            continue
        fw_file.write(line+"\n")



