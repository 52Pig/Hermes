import json

file1 = "D:/tool/dataset/strategy_data/std_reverse_moving_average_bull_track_20250604.json"
std = json.load(open(file1))

file2 = "D:/tool/dataset/strategy_data/reverse_moving_average_bull_track_20250604.json"
dt = json.load(open(file2))

std_list = list()
for k, v in std.items():
     is_target = v.get("is_target", "0")
     code = v.get("code", "0")

     if is_target == "1":
         std_list.append(code)


for k, v in dt.items():
     is_target = v.get("is_target", "0")
     if is_target != "1":
         continue
     code = v.get("code", "0")
     if code not in std_list:
         print(code)


