import json
from xtquant import xtdata

def f(data):
    print(json.dumps(data))
    slist = ['603887.SH', '002488.SZ', '002130.SZ', '002456.SZ', '600939.SH', '002366.SZ', '002583.SZ',
                            '600839.SH']
    for ele in slist:
        if ele not in data:
            print("error=", ele)

# clist = ["SH", "SZ"]
#clist = ["000560.SZ"]
subscribe_whole_list=['603887.SH', '002488.SZ', '002130.SZ', '002456.SZ', '600939.SH', '002366.SZ', '002583.SZ', '600839.SH']
xtdata.subscribe_whole_quote(code_list=subscribe_whole_list, callback=f)
xtdata.run()