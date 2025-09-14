import json
from xtquant import xtdata

slist = ['001317.SZ']
stock_code = '001317.SZ'
print("download tick start! recall_stock_size=", len(slist))

def callback(data):
    print(data)

def download_1():
    is_suc = xtdata.download_history_data2(slist, period='tick', start_time='20240901')
    # xtdata.subscribe_whole_quote(slist, callback=callback)
    # xtdata.run()
    data = xtdata.get_market_data_ex(
        stock_list=slist,
        period='tick',
        start_time='20240901',
        # start_time = '20240901093000'
    )
    # data = xtdata.get_local_data(
    #     stock_list=slist,
    #     period='tick',
    #     start_time='20240901'
    # )
    print(f'tick data={data[stock_code].columns}')
    print(f'tick data={json.dumps(data[stock_code].tail(2).to_dict())}')

def download_2():
    # is_suc = xtdata.download_history_data(stock_code, 'tick', '20240901')
    is_suc = xtdata.download_history_data(stock_code, 'tick', '20241213')
    print(f"download tick status={is_suc}")


    data = xtdata.get_market_data_ex(
        stock_list=slist,
        period='tick',
        start_time='20241213',
        # start_time = '20240901093000'
    )
    print(f'tick data={data[stock_code].columns}')
    print(f'tick data={json.dumps(data[stock_code].tail(2).to_dict())}')


download_1()
# download_2()



"""
{
    "time": {
        "20241213145951": 1734073191000,
        "20241213150000": 1734073200000
    },
    "lastPrice": {
        "20241213145951": 12.23,
        "20241213150000": 12.23
    },
    "open": {
        "20241213145951": 10.950000000000001,
        "20241213150000": 10.950000000000001
    },
    "high": {
        "20241213145951": 12.23,
        "20241213150000": 12.23
    },
    "low": {
        "20241213145951": 10.9,
        "20241213150000": 10.9
    },
    "lastClose": {
        "20241213145951": 11.120000000000001,
        "20241213150000": 11.120000000000001
    },
    "amount": {
        "20241213145951": 6550135230,
        "20241213150000": 6558021513
    },
    "volume": {
        "20241213145951": 5584136,
        "20241213150000": 5590585
    },
    "pvolume": {
        "20241213145951": 0,
        "20241213150000": 0
    },
    "stockStatus": {
        "20241213145951": 0,
        "20241213150000": 0
    },
    "openInt": {
        "20241213145951": 18,
        "20241213150000": 15
    },
    "lastSettlementPrice": {
        "20241213145951": 0,
        "20241213150000": 0
    },
    "askPrice": {
        "20241213145951": [
            12.23,
            0,
            0,
            0,
            0
        ],
        "20241213150000": [
            0,
            0,
            0,
            0,
            0
        ]
    },
    "bidPrice": {
        "20241213145951": [
            12.23,
            12.23,
            0,
            0,
            0
        ],
        "20241213150000": [
            12.23,
            12.23,
            12.23,
            12.23,
            12.23
        ]
    },
    "askVol": {
        "20241213145951": [
            6366,
            0,
            0,
            0,
            0
        ],
        "20241213150000": [
            0,
            0,
            0,
            0,
            0
        ]
    },
    "bidVol": {
        "20241213145951": [
            6366,
            326019,
            0,
            0,
            0
        ],
        "20241213150000": [
            327067,
            2963,
            348,
            1096,
            209
        ]
    },
    "settlementPrice": {
        "20241213145951": 0,
        "20241213150000": 0
    },
    "transactionNum": {
        "20241213145951": 329651,
        "20241213150000": 329938
    },
    "pe": {
        "20241213145951": 0,
        "20241213150000": 0
    }
}  
"""