import json
from xtquant import xtdata

def f(data):
    print(json.dumps(data))
    # slist = ['603887.SH', '002488.SZ', '002130.SZ', '002456.SZ', '600939.SH', '002366.SZ', '002583.SZ',
    #                         '600839.SH']
    # for ele in slist:
    #     if ele not in data:
    #         print("error=", ele)

# clist = ["SH", "SZ"]
#clist = ["000560.SZ"]
#subscribe_whole_list=['603887.SH', '002488.SZ', '002130.SZ', '002456.SZ', '600939.SH', '002366.SZ', '002583.SZ', '600839.SH']
subscribe_whole_list=['603776.SH']
xtdata.subscribe_whole_quote(code_list=subscribe_whole_list, callback=f)
xtdata.run()



'''2024-12-06
{
    "603776.SH": {
        "time": 1733468404000,
        "lastPrice": 19.02,    //最新价格
        "open": 19.02,         //开盘价
        "high": 19.02,         //最高
        "low": 19.02,          //最低
        "lastClose": 17.29,    //昨收
        "amount": 5476300,     //成交金额
        "volume": 2879,        //成交量
        "pvolume": 287922,     //总共多少手
        "stockStatus": 0,      //
        "openInt": 15,         //股票状态
        "transactionNum": 0,      //成交笔数
        "lastSettlementPrice": 0,  //前结算(股票为0)
        "settlementPrice": 0,      //今结算(股票为0)
        "pe": 0,                   //
        "askPrice": [              //多档委卖价
            0,
            0,
            0,
            0,
            0
        ],
        "bidPrice": [             //多档委买价
            19.02,
            19.01,
            19,
            0,
            0
        ],
        "askVol": [             //多档委卖量
            0,
            0,
            0,
            0,
            0
        ],
        "bidVol": [            //多档委买量
            146392,
            826,
            639,
            0,
            0
        ],
        "volRatio": 0,       //成交量比例
        "speed1Min": 0,      // 1分钟涨速
        "speed5Min": 0       // 5分钟涨速
    }
}   '''

