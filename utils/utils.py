
import sys
import time
import random
from collections import OrderedDict
from tqdm import tqdm
import re
from requests.adapters import Retry, HTTPAdapter
import pandas_market_calendars as mcal
sys.path.append('../')
import numpy as np
from datetime import datetime, timedelta, date
from xtquant import xtdata
import requests
import json
import akshare as ak
import pandas as pd
# import tdx_data

# 行业代码映射表
industry_mapping = {
    "1": "金融行业",
    "2": "酿酒行业",
    "3": "电子信息",
    "4": "医药制造",
    "5": "汽车行业",
    "6": "房地产",
    "7": "化工行业",
    "8": "机械行业",
    "9": "电力行业",
    "10": "有色金属",
    # ... 其他行业代码需根据实际响应补充
    "881155": "白酒",          # 示例特殊行业代码
    "881120": "半导体"         # 示例特殊行业代码
}

def get_a_share_trade_days_from_ak(start_date='20250601', end_date=None):
    """
    从AKShare获取A股真实交易日历（解决节假日问题）
    """
    # 获取全量交易日历
    trade_days_all = ak.tool_trade_date_hist_sina()
    # 统一转换为date对象
    def to_date(d):
        if isinstance(d, str):
            return datetime.strptime(d, "%Y%m%d").date()
        elif isinstance(d, date):
            return d
        raise ValueError("日期格式应为'YYYYMMDD'字符串或date对象")

    start_date = to_date(start_date)
    end_date = to_date(end_date) if end_date else datetime.now().date()

    # 筛选日期范围
    trade_days = [
        day.strftime("%Y%m%d")
        for day in trade_days_all["trade_date"]
        if start_date <= day <= end_date
    ]
    return sorted(trade_days)

def get_a_share_trade_days(start_date='20240508', end_date=None):
    """
    获取A股所有交易日（从 start_date 到 end_date，默认到今日）
    :param start_date: 起始日期（格式 'YYYYMMDD'）
    :param end_date: 结束日期（格式 'YYYYMMDD'，默认今日）
    :return: 交易日列表（格式 ['YYYYMMDD', ...]）
    """
    # 创建上交所日历对象
    sse_calendar = mcal.get_calendar('XSHG')  # XSHG 代表上海证券交易所

    # 日期格式转换
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp(datetime.now().date())

    # 获取交易日范围
    schedule = sse_calendar.schedule(start_date=start, end_date=end)

    # 生成交易日列表（格式化为字符串）
    # trade_days = set([
    #     day.strftime('%Y%m%d')
    #     for day in mcal.date_range(schedule, frequency='1D')
    # ])
    trade_days = [day.strftime('%Y%m%d') for day in schedule.index.date]
    trade_days = sorted(trade_days)
    return trade_days


def to_onehot(value, num_classes):
    """
    将数值转换为One-Hot编码。

    参数:
    value -- 要转换的数值
    num_classes -- One-Hot编码的维度（类别总数）

    返回:
    onehot_encoded -- One-Hot编码的数组
    """
    # 创建一个全0的数组，长度为One-Hot编码的维度
    onehot_encoded = np.zeros(num_classes)

    # 将对应数值的索引位置设为1
    if 0 <= value < num_classes:
        onehot_encoded[value] = 1
    else:
        raise ValueError(f"Value {value} is out of range for {num_classes} classes.")
    return onehot_encoded

def parse_time(time_str):
    ''' 根据时间获取处理后的时间特征
    :param time_str:
    :return:
    '''
    # 解析时间字符串
    timestamp = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S+08:00')
    dt = datetime.strftime(timestamp, '%Y-%m-%d %H:%M:%S')
    # print(dt, time_str)
    # 计算所需的时间单位信息
    year_day = timestamp.timetuple().tm_yday  # 一年中的第几天
    year_month = timestamp.month  # 一年中的第几月
    month_day = timestamp.day  # 一月中的第几天
    week_day = timestamp.weekday() + 1  # 一星期中的第几天（星期一为1）
    hour_of_day = timestamp.hour  # 一天中的第几小时
    minute_of_hour = timestamp.minute  # 一小时中的第几分钟

    return dt, year_day, year_month, month_day, week_day, hour_of_day, minute_of_hour

def get_latest_price(stock_code, is_download=False):
    """# 定义获取股票当前价格的函数
    """
    if is_download:
        xtdata.download_history_data(stock_code, '1m', '20240901')
    data = xtdata.get_market_data_ex(
        ['time', 'close'],
        stock_list=[stock_code],
        period='1m',
        start_time='20240808093000',dividend_type="front")
    if data is None:
        return None
    # print('[utils]get_latest_price, data=', data)
    df = data[stock_code]
    if df is None or df.empty:
        return None
    # 获取最大时间戳的行
    tdata = df.loc[df['time'].idxmax()]
    # print('[utils]get_latest_price, tdata=', tdata)
    if 'close' in tdata:
        return tdata['close']
    else:
        return None

def get_current_time():
    return datetime.now().strftime('%Y%m%d %H:%M:%S')

def get_current_date():
    return datetime.now().strftime('%Y%m%d')

def get_past_date(days):
    # 计算过去的日期
    current_date = datetime.now()
    past_date = current_date - timedelta(days=days)
    return past_date.date()

def get_past_trade_date(n):
    """ 获取过去N个交易日
    """
    import a_trade_calendar
    # 获取今天的日期
    today = a_trade_calendar.get_latest_trade_date()
    # 计算过去N个交易日的日期
    # 假设我们要查询过去5个交易日
    trade_dates = a_trade_calendar.get_pre_trade_date(today, n)
    print("[utils]get_past_trade_date:today=", today, ";trade_dates=", trade_dates)
    return trade_dates

def get_yesterday_close_price(stock_code):
    """获取股票昨日收盘价格的函数"""
    data = xtdata.get_market_data_ex(
        stock_list=[stock_code],
        field_list=['time', 'close'],
        period='1d',
        count=2,  # 获取最近两天的数据
        dividend_type = "front"
    )
    if data is None:
        return None
    df = data[stock_code]
    if df is None or df.empty:
        return None
    if len(df) < 2:
        return None  # 不足两天数据，返回None
    # 获取前一天的收盘价格
    yesterday_close = df.iloc[-2]['close']  # 倒数第二行是昨日的收盘价
    return yesterday_close

def get_close_price(stock_code, last_n=1):
    """获取股票过去第 n 天收盘价格的函数"""
    # 获取最近 last_n+1 天的数据
    data = xtdata.get_market_data_ex(
        stock_list=[stock_code],
        field_list=['time', 'close'],
        period='1d',
        count=last_n + 1,
        dividend_type = "front"
    )
    if data is None:
        return None
    df = data[stock_code]
    if df is None or df.empty:
        return None
    if len(df) < last_n + 1:
        return None  # 数据不足，返回 None
    # 获取过去第 n 天的收盘价格
    close_price = df.iloc[-(last_n + 1)]['close']
    return close_price

def get_stock_hist_minutes_data_em_with_retry(stock='600031', start_date='20210101', end_date='20500101', data_type='D', count=8000):
    """
    获取分钟K线数据

    参数:
    secid : str
        股票代码（格式：市场代码.股票代码，如 1.600000 表示上证600000）
    klt : int
        K线周期（分钟数，如1 5 15 30 60）
    lmt : int
        获取数据条数（最大10000）
    """
    marker, stock = rename_stock_type_1(stock)
    secid = f'{marker}.{stock}'
    klt = 1
    lmt = 10000
    url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5",
        "fields2": "f51,f52,f53,f54,f55,f56,f57",
        "klt": klt,
        "lmt": lmt,
        "end": "20500101",  # 结束时间设为未来日期表示获取最新数据
        "fqt": "1"
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    try:
        print(f"request={params}")
        response = requests.get(url, params=params, headers=headers)
        data = response.json()

        if data.get("data") is None:
            print("未获取到数据，请检查参数")
            return None

        klines = data["data"]["klines"]
        result = []

        for k in klines:
            items = k.split(",")
            result.append({
                "时间": items[0],
                "开盘": float(items[1]),
                "收盘": float(items[2]),
                "最高": float(items[3]),
                "最低": float(items[4]),
                "成交量": int(items[5]),
                "成交额": float(items[6]),
                # "换手率": float(items[7])
            })

        df = pd.DataFrame(result)
        return df

    except Exception as e:
        print(f"获取数据失败：{e}")
        return None


def get_stock_hist_data_em_with_retry(stock='600031', start_date='20210101', end_date='20500101', data_type='D', count=8000):
    '''
    获取股票数据（集成超时重试机制）
    start_date=''默认上市时间
    - ``1`` : 分钟
        - ``5`` : 5 分钟
        - ``15`` : 15 分钟
        - ``30`` : 30 分钟
        - ``60`` : 60 分钟
        - ``101`` : 日
        - ``102`` : 周
        - ``103`` : 月
    fq=0股票除权
    fq=1前复权
    fq=2后复权
    '''
    # 配置请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Referer': 'http://push2his.eastmoney.com/'
    }
    # print(f"start={start_date},end={end_date}")

    # 创建带重试机制的Session
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=['GET']
    )
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    try:
        data_dict = {'1': '1', '5': '5', '15': '15', '30': '30', '60': '60', 'D': '101', 'W': '102', 'M': '103'}
        klt = data_dict[data_type]
        marker, stock = rename_stock_type_1(stock)
        secid = f'{marker}.{stock}'

        url = 'http://push2his.eastmoney.com/api/qt/stock/kline/get?'
        params = {
            'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
            'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
            # "fields1": "f1,f2,f3,f4,f5",
            # "fields2": "f51,f52,f53,f54,f55,f56,f57",

            'beg': start_date,
            'end': end_date,
            'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
            # 'rtntype': '6',  # 注意：此参数疑似错误，正常应为数据类型
            'secid': secid,
            'klt': int(klt),
            'fqt': '1',  # 前复权
            'lmt': count,
            'cb': 'jsonp1668432946680'
        }

        # 发送请求（连接超时10s，读取超时30s）
        # time.sleep(min(random.random(), 0.6))
        time.sleep(1)
        res = session.get(url, params=params, headers=headers, timeout=(30, 30))
        res.raise_for_status()  # 自动处理HTTP错误
        # print(res.text)
        # 数据处理逻辑
        # text = re.findall(r'\((.*?)\)', res.text)[0]  # 提取括号内的内容
        text = res.text[19:len(res.text) - 2]
        json_text = json.loads(text)

        df = pd.DataFrame(json_text['data']['klines'])
        try:
            df.columns = ['数据']
        except Exception as e:
            print(f"[ERROR] 数据结构异常 code={stock}: {str(e)}")
            return pd.DataFrame()

        data_list = [i.split(',') for i in df['数据']]
        data = pd.DataFrame(data_list)
        columns = ['date', 'open', 'close', 'high', 'low', 'volume', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
        data.columns = columns

        # 类型转换
        for m in columns[1:]:
            data[m] = pd.to_numeric(data[m], errors='coerce')

        return data.sort_index(ascending=True, ignore_index=True)

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 请求失败: {str(e)}")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON解析失败: {str(e)}")
        return pd.DataFrame()
    except KeyError as e:
        print(f"[ERROR] 数据字段缺失: {str(e)}")
        return pd.DataFrame()
#
# import requests
# from requests.adapters import HTTPAdapter
# from requests.packages.urllib3.util.retry import Retry
# import json
# import pandas as pd
import re
# from datetime import datetime


from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def get_data_via_browser(stock_code):
    options = Options()
    options.headless = True  # 无头模式
    driver = webdriver.Chrome(options=options)
    url = f'https://quote.eastmoney.com/sh{stock_code}.html'
    driver.get(url)
    kline_data = driver.execute_script('return window.klineData')  # 假设数据在全局变量中
    driver.quit()
    return pd.DataFrame(kline_data)

def get_ut_token():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.eastmoney.com/",
        "Cookie": "qgqp_b_id=3b7d8...; other_cookie_key=value",  # 必须从浏览器复制
        "Connection": "keep-alive"
    }
    home_url = "https://www.eastmoney.com/"
    res = requests.get(home_url, headers=headers)
    print(res.text)
    # 使用正则或BeautifulSoup从页面中提取ut值
    # ut = re.search(r'"ut":"(\w+)"', res.text).group(1)
    ut=""
    return ut

def get_stock_hist_data_em_with_retry_v2(stock='600031', start_date='20210101', end_date=None, data_type='D', count=8000):
    '''
    获取股票数据（集成超时重试机制）
    - data_type映射关系:
        ``1`` : 分钟, ``5`` : 5分钟, ``15`` : 15分钟, ``30`` : 30分钟, ``60`` : 60分钟,
        ``D`` : 日(101), ``W`` : 周(102), ``M`` : 月(103)
    - fqt=1 表示前复权
    '''
    # 动态设置end_date为当前日期
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    headers = {
        # 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        # 'Referer': 'https://www.eastmoney.com/',

        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:137.0) Gecko/20100101 Firefox/137.0",
        "Referer": "https://www.eastmoney.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests":"1",
        "Host":"push2his.eastmoney.com",
        # "Cookie":"HAList=ty-1-600859-%u738B%u5E9C%u4E95; st_si=20433206087918; st_sn=7; st_psi=20250421232719267-113200301201-6687692229; st_asi=delete; qgqp_b_id=ab0fc0964e268ffc6725db04699f0bee; fullscreengg=1; fullscreengg2=1; st_pvi=90574784602420; st_sp=2025-04-21%2022%3A15%3A02; st_inirUrl=https%3A%2F%2Fquote.eastmoney.com%2Fsh600859.html",

        # "Cookie": "ab0fc0964e268ffc6725db04699f0bee"
    }

    # 配置带重试的Session
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=['GET'],
        respect_retry_after_header=False
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))

    # try:
    data_dict = {'1':'1', '5':'5', '15':'15', '30':'30', '60':'60',
                'D':'101', 'W':'102', 'M':'103'}
    klt = data_dict.get(data_type, '101')  # 默认日线
    marker, stock_code = rename_stock_type_1(stock)  # 确保此函数正确生成secid
    secid = f'{marker}.{stock_code}'

    url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    params = {
        'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'beg': start_date,
        'end': end_date,
        'ut': 'fa5fd1943c7b386f172d6893dbfba10b',  # 检查是否需要更新
        # 'ut': 'ab0fc0964e268ffc6725db04699f0bee',
        'rtntype': '6',  # 固定为6返回标准JSON
        'secid': secid,
        'klt': klt,
        'fqt': '1',
        'cb': 'jsonpCallback'
    }

    # proxies = {"http": "http://10.10.1.10:3128", "https": "http://10.10.1.10:1080"}
    res = session.get(url, params=params, headers=headers, timeout=(10, 30))
    print(res)
    res.raise_for_status()

    # 正则提取JSON数据
    json_str = re.search(r'jsonpCallback\((.*?)\);', res.text)
    if not json_str:
        raise ValueError("响应格式无效，未找到JSON数据")
    json_data = json.loads(json_str.group(1))

    # 数据解析
    klines = json_data.get('data', {}).get('klines', [])
    if not klines:
        print(f"[WARN] 无数据: code={stock}")
        return pd.DataFrame()

    data = pd.DataFrame([k.split(',') for k in klines])
    columns = ['date', 'open', 'close', 'high', 'low', 'volume',
              '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
    data.columns = columns

    # 类型转换
    numeric_cols = columns[1:]
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

    return data.sort_values('date', ignore_index=True)

    # except Exception as e:
    #     print(f"[ERROR] 处理失败: {str(e)}")
    #     return pd.DataFrame()

def adjust_stock(stock='600031.SH'):
    '''
    调整代码
    '''
    if stock[-2:]=='SH' or stock[-2:]=='SZ' or stock[-2:]=='sh' or stock[-2:]=='sz':
        stock=stock.upper()
    else:
        if stock[:3] in ['110','113','118','510','519',
                        "900",'200'] or stock[:2] in ['11','51','60','68'] or stock[:1] in ['5']:
            stock=stock+'.SH'
        else:
            stock=stock+'.SZ'
    return stock


def get_index_data_em_from_qmt(stock='600031',start_date='20210101',end_date='20500101',data_type='D',count=8000):
    '''
    获取股票数据
    start_date=''默认上市时间
    - ``1`` : 分钟
        - ``5`` : 5 分钟
        - ``15`` : 15 分钟
        - ``30`` : 30 分钟
        - ``60`` : 60 分钟
        - ``101`` : 日
        - ``102`` : 周
        - ``103`` : 月
    fq=0股票除权
    fq=1前复权
    fq=2后复权
    '''
    #qmt数据
    qmt_dict={'1': '1m', '5': '5m', '15': '15m', '30m': '30m', '60': '60m', 'D': '1d', 'W': '1w', 'M': '1mon',
            "tick":'tick','1q':"1q","1hy":"1hy","1y":"1y"}
    data_dict = {'1': '1', '5': '5', '15': '15', '30': '30', '60': '60', 'D': '101', 'W': '102', 'M': '103'}
    stock_1 = stock
    period = qmt_dict.get(data_type,'D')
    # print("-----------1----------", stock_1)
    subno = xtdata.subscribe_quote(stock_code=stock_1,period=period,start_time=start_date,end_time=end_date,count=-1)
    data = xtdata.get_market_data_ex(stock_list=[stock_1],period=period,start_time=start_date, end_time=end_date,count=-1,dividend_type="front")
    # print("-----------------", data.values())
    data = data[stock_1]
    ## 关键，不取消会导致个别数据不全
    xtdata.unsubscribe_quote(subno)

    # print(data.columns)
    # ['time', 'open', 'high', 'low', 'close', 'volume', 'amount', 'settelementPrice', 'openInterest', 'preClose', 'suspendFlag']
    ## settelementPrice： 今结算
    ## openInterest: 持仓量
    ## preClose: 前收盘价
    ## suspendFlag：1停牌，0不停牌

    # print(data.head(20).index, data['volume'].head(20), data['amount'].head(20))
    # print(data.head(20).index, data['settelementPrice'].head(20))
    if data.shape[0]>0:
        ##schema：['date', 'open', 'close', 'high', 'low', 'volume', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
        # 获取流通股本（单位：股）
        instrument_info = xtdata.get_instrument_detail(stock_1)
        float_shares = 0
        if instrument_info is not None:
            float_shares = instrument_info.get("FloatVolume", 0)
        # 手动计算换手率：换手率 = 成交量 / 流通股本 * 100%
        if float_shares > 0:
            # data['换手率'] = data['volume'] / float_shares * 100  # 单位：百分比
            data['换手率'] = (data['volume'] * 100) / float_shares * 100  # 乘以100将手转为股
        else:
            data['换手率'] = 0  # 处理异常情况

        # print('-----------------', data)
        # 其他字段处理
        data['date'] = data.index.tolist()
        data['成交额'] = data['amount']
        data['振幅'] = data['high'] - data['low']

        data['涨跌幅'] = data['close'].pct_change().fillna(0) * 100  # 首日填充为0
        data['涨跌额'] = (data['close'] - data['close'].shift(1)).fillna(0)  # 首日填充为0
        # data['涨跌幅'] = data['close'].pct_change() * 100
        # data['涨跌额'] = data['close'] - data['close'].shift(1)
        return data
    else:
        return None


def get_batch_stock_data_em_from_qmt(stock_list=['600031'], start_date='20210101', end_date='20500101', data_type='D',
                                     count=8000):
    '''
    获取股票数据（支持单个股票或股票列表）
    stock: 单个股票代码或股票代码列表
    start_date: 开始日期（默认上市时间）
    end_date: 结束日期
    data_type: 数据类型
        - ``1`` : 分钟
        - ``5`` : 5 分钟
        - ``15`` : 15 分钟
        - ``30`` : 30 分钟
        - ``60`` : 60 分钟
        - ``101`` : 日
        - ``102`` : 周
        - ``103`` : 月
    count: 最大数据条数
    '''
    qmt_dict = {'1': '1m', '5': '5m', '15': '15m', '30m': '30m', '60': '60m', 'D': '1d', 'W': '1w', 'M': '1mon',
                "tick": 'tick', "1q": "1q", "1hy": "1hy", "1y": "1y"}
    data_dict = {'1': '1', '5': '5', '15': '15', '30': '30', '60': '60', 'D': '101', 'W': '102', 'M': '103'}
    period = qmt_dict.get(data_type, 'D')

    # 统一处理为股票列表
    # stock_list = [stock] if isinstance(stock, str) else stock
    adjusted_stocks = [adjust_stock(s) for s in stock_list]

    # 分批处理（每批50支）
    batch_size = 50
    result_dict = {}

    for i in range(0, len(adjusted_stocks), batch_size):
        print(f"process batch_i={i}, all_code_size={len(stock_list)}, batch_size={batch_size}")
        batch = adjusted_stocks[i:i + batch_size]

        # 订阅当前批次
        subno_list = list()
        for stock in batch:
            subno = xtdata.subscribe_quote(
                stock_code=stock,
                period=period,
                start_time=start_date,
                end_time=end_date,
                count=-1
            )
            subno_list.append(subno)

        # 获取数据
        batch_data = xtdata.get_market_data_ex(
            stock_list=batch,
            period=period,
            start_time=start_date,
            end_time=end_date,
            count=-1,
            dividend_type="front"
        )

        # 立即取消订阅当前批次
        for subno in subno_list:
            xtdata.unsubscribe_quote(subno)
        subno_list.clear()
        # 处理每支股票的数据
        for stock_code in batch:
            data = batch_data.get(stock_code)
            if data is None or data.shape[0] == 0:
                continue

            # 获取流通股本
            instrument_info = xtdata.get_instrument_detail(stock_code)
            float_shares = instrument_info.get("FloatVolume", 0) if instrument_info else 0

            # 计算换手率
            if float_shares > 0:
                data['换手率'] = (data['volume'] * 100) / float_shares * 100
            else:
                data['换手率'] = 0

            # 添加其他字段
            data['date'] = data.index
            data['成交额'] = data['amount']
            data['振幅'] = data['high'] - data['low']
            data['涨跌幅'] = data['close'].pct_change().fillna(0) * 100
            data['涨跌额'] = (data['close'] - data['close'].shift(1)).fillna(0)

            # 保存处理后的数据
            original_code = stock_list[adjusted_stocks.index(stock_code)]  # 获取原始股票代码
            result_dict[original_code] = data

    # 返回单个DataFrame（输入为单个股票）或字典（输入为列表）
    return result_dict

def get_stock_today_data_em_from_qmt(stock='600031',start_date='20210101',end_date='20500101',data_type='D',count=8000):
    '''
    获取股票数据
    start_date=''默认上市时间
    - ``1`` : 分钟
        - ``5`` : 5 分钟
        - ``15`` : 15 分钟
        - ``30`` : 30 分钟
        - ``60`` : 60 分钟
        - ``101`` : 日
        - ``102`` : 周
        - ``103`` : 月
    fq=0股票除权
    fq=1前复权
    fq=2后复权
    '''
    #qmt数据
    qmt_dict={'1': '1m', '5': '5m', '15': '15m', '30m': '30m', '60': '60m', 'D': '1d', 'W': '1w', 'M': '1mon',
            "tick":'tick','1q':"1q","1hy":"1hy","1y":"1y"}
    data_dict = {'1': '1', '5': '5', '15': '15', '30': '30', '60': '60', 'D': '101', 'W': '102', 'M': '103'}
    stock_1 = adjust_stock(stock=stock)
    period = qmt_dict.get(data_type,'D')
    subno = xtdata.subscribe_quote(stock_code=stock_1,period=period,start_time=start_date,end_time=end_date,count=-1)
    data = xtdata.get_market_data_ex(stock_list=[stock_1],period=period,start_time=start_date, end_time=end_date,count=-1,dividend_type="front")
    # print("-----------------", data.values())
    data = data[stock_1]
    ## 关键，不取消会导致个别数据不全
    xtdata.unsubscribe_quote(subno)

    # print(data.columns)
    # ['time', 'open', 'high', 'low', 'close', 'volume', 'amount', 'settelementPrice', 'openInterest', 'preClose', 'suspendFlag']
    ## settelementPrice： 今结算
    ## openInterest: 持仓量
    ## preClose: 前收盘价
    ## suspendFlag：1停牌，0不停牌

    # print(data.head(20).index, data['volume'].head(20), data['amount'].head(20))
    # print(data.head(20).index, data['settelementPrice'].head(20))
    if data.shape[0]>0:
        ##schema：['date', 'open', 'close', 'high', 'low', 'volume', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
        # 获取流通股本（单位：股）
        instrument_info = xtdata.get_instrument_detail(stock_1)
        float_shares = 0
        if instrument_info is not None:
            float_shares = instrument_info.get("FloatVolume", 0)
        # 手动计算换手率：换手率 = 成交量 / 流通股本 * 100%
        if float_shares > 0:
            # data['换手率'] = data['volume'] / float_shares * 100  # 单位：百分比
            data['换手率'] = (data['volume'] * 100) / float_shares * 100  # 乘以100将手转为股
        else:
            data['换手率'] = 0  # 处理异常情况

        # 其他字段处理
        data['date'] = data.index.tolist()
        data['成交额'] = data['amount']
        data['振幅'] = data['high'] - data['low']

        data['涨跌幅'] = data['close'].pct_change().fillna(0) * 100  # 首日填充为0
        data['涨跌额'] = (data['close'] - data['close'].shift(1)).fillna(0)  # 首日填充为0
        # data['涨跌幅'] = data['close'].pct_change() * 100
        # data['涨跌额'] = data['close'] - data['close'].shift(1)
        return data
    else:
        return None



def get_stock_hist_data_em_from_qmt(stock='600031',start_date='20210101',end_date='20500101',data_type='D',count=8000):
    '''
    获取股票数据
    start_date=''默认上市时间
    - ``1`` : 分钟
        - ``5`` : 5 分钟
        - ``15`` : 15 分钟
        - ``30`` : 30 分钟
        - ``60`` : 60 分钟
        - ``101`` : 日
        - ``102`` : 周
        - ``103`` : 月
    fq=0股票除权
    fq=1前复权
    fq=2后复权
    '''
    #qmt数据
    qmt_dict={'1': '1m', '5': '5m', '15': '15m', '30m': '30m', '60': '60m', 'D': '1d', 'W': '1w', 'M': '1mon',
            "tick":'tick','1q':"1q","1hy":"1hy","1y":"1y"}
    data_dict = {'1': '1', '5': '5', '15': '15', '30': '30', '60': '60', 'D': '101', 'W': '102', 'M': '103'}
    stock_1 = adjust_stock(stock=stock)
    period = qmt_dict.get(data_type,'D')
    xtdata.subscribe_quote(stock_code=stock_1,period=period,start_time=start_date,end_time=end_date,count=-1)
    data = xtdata.get_market_data_ex(stock_list=[stock_1],period=period,start_time=start_date,end_time=end_date,count=-1,dividend_type="front")
    # print("-----------------", data)
    data = data[stock_1]
    # print(data.columns)
    # ['time', 'open', 'high', 'low', 'close', 'volume', 'amount', 'settelementPrice', 'openInterest', 'preClose', 'suspendFlag']
    ## settelementPrice： 今结算
    ## openInterest: 持仓量
    ## preClose: 前收盘价
    ## suspendFlag：1停牌，0不停牌

    # print(data.head(20).index, data['volume'].head(20), data['amount'].head(20))
    # print(data.head(20).index, data['settelementPrice'].head(20))
    if data.shape[0]>0:
        ##schema：['date', 'open', 'close', 'high', 'low', 'volume', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
        # 获取流通股本（单位：股）
        instrument_info = xtdata.get_instrument_detail(stock_1)
        float_shares = 0
        if instrument_info is not None:
            float_shares = instrument_info.get("FloatVolume", 0)
        # 手动计算换手率：换手率 = 成交量 / 流通股本 * 100%
        if float_shares > 0:
            # data['换手率'] = data['volume'] / float_shares * 100  # 单位：百分比
            data['换手率'] = (data['volume'] * 100) / float_shares * 100  # 乘以100将手转为股
        else:
            data['换手率'] = 0  # 处理异常情况

        # 其他字段处理
        data['date'] = data.index.tolist()
        data['成交额'] = data['amount']
        data['振幅'] = data['high'] - data['low']

        data['涨跌幅'] = data['close'].pct_change().fillna(0) * 100  # 首日填充为0
        data['涨跌额'] = (data['close'] - data['close'].shift(1)).fillna(0)  # 首日填充为0
        # data['涨跌幅'] = data['close'].pct_change() * 100
        # data['涨跌额'] = data['close'] - data['close'].shift(1)
        return data
    else:
        return None

def get_stock_hist_data_em(stock='600031', start_date='20210101', end_date='20500101', data_type='D', count=8000):
    '''
    获取股票数据
    start_date=''默认上市时间
    - ``1`` : 分钟
        - ``5`` : 5 分钟
        - ``15`` : 15 分钟
        - ``30`` : 30 分钟
        - ``60`` : 60 分钟
        - ``101`` : 日
        - ``102`` : 周
        - ``103`` : 月
    fq=0股票除权
    fq=1前复权
    fq=2后复权
    '''
    data_dict = {'1': '1', '5': '5', '15': '15', '30': '30', '60': '60', 'D': '101', 'W': '102', 'M': '103'}
    klt = data_dict[data_type]
    marker, stock = rename_stock_type_1(stock)
    secid = '{}.{}'.format(marker, stock)
    # print(f'secid={secid}')
    url = 'http://push2his.eastmoney.com/api/qt/stock/kline/get?'
    params = {
        'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'beg': start_date,
        'end': end_date,
        'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
        'rtntype': end_date,
        'secid': secid,
        'klt': klt,
        'fqt': '1',   #前复权
        'cb': 'jsonp1668432946680'
    }
#    try:
    # 请求超时，读取超时，单位秒
    res = requests.get(url=url, params=params, timeout=(5, 5))
    res.raise_for_status()  # 自动处理4xx/5xx错误

    text = res.text[19:len(res.text) - 2]
    json_text = json.loads(text)
    # print(json_text)
    df = pd.DataFrame(json_text['data']['klines'])
    try:
        df.columns = ['数据']
    except:
        print(f"[utils]get_stock_hist_data_em error code={stock}")
        return pd.DataFrame({})
    data_list = []
    for i in df['数据']:
        data_list.append(i.split(','))
    data = pd.DataFrame(data_list)
    columns = ['date', 'open', 'close', 'high', 'low', 'volume', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
    data.columns = columns
    for m in columns[1:]:
        data[m] = pd.to_numeric(data[m])
    data.sort_index(ascending=True, ignore_index=True, inplace=True)
    return data
#    except:
#         '''
#         0 5分钟K线
#         1 15分钟K线
#         2 30分钟K线
#         3 1小时K线
#         4 日K线
#         7 1分钟
#         8 1分钟K线
#         '''
        # data_list = []
        # data = pd.DataFrame(data_list)
        # print(f'[utils][get_stock_hist_data_em]{stock} interface return null!')
        # return data




        # data_dict = {'1': '8', '5': '0', '15': '1', '30': '2', '60': '3', 'D': '4', '7': '7', }
        # n = data_dict[data_type]
        # data = get_security_minute_data(stock=stock, n=n, count=count)
        # data['date'] = data['datetime'].apply(lambda x: str(x)[:10])
        # data['volume'] = data['vol']
        # data['涨跌幅'] = data['close'].pct_change() * 100
        # data['涨跌额'] = data['close'] - data['open']
        # data['振幅'] = (data['high'] - data['low']) / data['low'] * 100
        # return data

def get_stock_now_data(code_list=['600031']):
    """
    东方财富网-沪深京 A 股-实时行情
    https://quote.eastmoney.com/center/gridlist.html#hs_a_board
    :return: 实时行情
    :rtype: pandas.DataFrame
    """
    """获取沪深京全部A股实时行情数据（修复NoneType错误）"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:91.0) Gecko/20100101 Firefox/91.0'
    }
    url = "http://82.push2.eastmoney.com/api/qt/clist/get"

    all_data = []
    max_retries = 1  # 最大重试次数

    for page in range(1, 1000):  # 假设最多有1000页（根据实际情况调整）
        params = {
            "pn": str(page),
            "pz": "100",  # 明确设置为100（观察服务器实际支持的单页最大值）
            "po": "1",
            "np": "1",
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": "2",
            "invt": "2",
            "fid": "f3",
            "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
            "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152",
            "_": "1623833739532",
        }

        # 添加重试机制
        for _ in range(max_retries):
            try:
                r = requests.get(url, params=params, headers=headers, timeout=(30,30))
                data_json = r.json()
                if not data_json["data"]["diff"]:
                    break  # 数据已全部获取，退出循环
                temp_df = pd.DataFrame(data_json["data"]["diff"])
                all_data.append(temp_df)
                # print(f"成功获取第 {page} 页，共 {len(temp_df)} 条数据")
                break
            except Exception as e:
                print(f"第 {page} 页请求失败，重试中... 错误：{e}")
                time.sleep(2)  # 等待2秒后重试
        else:
            print(f"第 {page} 页请求失败，已达最大重试次数")
            break

        # 如果当前页数据量小于 pz，说明已到最后一页
        if len(temp_df) < int(params["pz"]):
            break

    if not all_data:
        return pd.DataFrame()

    temp_df = pd.concat(all_data, ignore_index=True)
    # if not all_data:
    #     return pd.DataFrame()

    # temp_df = pd.concat(all_data, ignore_index=True)

    temp_df.columns = [
        "_",
        "最新价",
        "涨跌幅",
        "涨跌额",
        "成交量",
        "成交额",
        "振幅",
        "换手率",
        "市盈率-动态",
        "量比",
        "5分钟涨跌",
        "代码",
        "_",
        "名称",
        "最高",
        "最低",
        "今开",
        "昨收",
        "总市值",
        "流通市值",
        "涨速",
        "市净率",
        "60日涨跌幅",
        "年初至今涨跌幅",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]
    temp_df.reset_index(inplace=True)
    temp_df["index"] = temp_df.index + 1
    temp_df.rename(columns={"index": "序号"}, inplace=True)
    temp_df = temp_df[
        [
            "序号",
            "代码",
            "名称",
            "最新价",
            "涨跌幅",
            "涨跌额",
            "成交量",
            "成交额",
            "振幅",
            "最高",
            "最低",
            "今开",
            "昨收",
            "量比",
            "换手率",
            "市盈率-动态",
            "市净率",
            "总市值",
            "流通市值",
            "涨速",
            "5分钟涨跌",
            "60日涨跌幅",
            "年初至今涨跌幅",
        ]
    ]
    temp_df["最新价"] = pd.to_numeric(temp_df["最新价"], errors="coerce")
    temp_df["涨跌幅"] = pd.to_numeric(temp_df["涨跌幅"], errors="coerce")
    temp_df["涨跌额"] = pd.to_numeric(temp_df["涨跌额"], errors="coerce")
    temp_df["成交量"] = pd.to_numeric(temp_df["成交量"], errors="coerce")
    temp_df["成交额"] = pd.to_numeric(temp_df["成交额"], errors="coerce")
    temp_df["振幅"] = pd.to_numeric(temp_df["振幅"], errors="coerce")
    temp_df["最高"] = pd.to_numeric(temp_df["最高"], errors="coerce")
    temp_df["最低"] = pd.to_numeric(temp_df["最低"], errors="coerce")
    temp_df["今开"] = pd.to_numeric(temp_df["今开"], errors="coerce")
    temp_df["昨收"] = pd.to_numeric(temp_df["昨收"], errors="coerce")
    temp_df["量比"] = pd.to_numeric(temp_df["量比"], errors="coerce")
    temp_df["换手率"] = pd.to_numeric(temp_df["换手率"], errors="coerce")
    temp_df["市盈率-动态"] = pd.to_numeric(temp_df["市盈率-动态"], errors="coerce")
    temp_df["市净率"] = pd.to_numeric(temp_df["市净率"], errors="coerce")
    temp_df["总市值"] = pd.to_numeric(temp_df["总市值"], errors="coerce")
    temp_df["流通市值"] = pd.to_numeric(temp_df["流通市值"], errors="coerce")
    temp_df["涨速"] = pd.to_numeric(temp_df["涨速"], errors="coerce")
    temp_df["5分钟涨跌"] = pd.to_numeric(temp_df["5分钟涨跌"], errors="coerce")
    temp_df["60日涨跌幅"] = pd.to_numeric(temp_df["60日涨跌幅"], errors="coerce")
    temp_df["年初至今涨跌幅"] = pd.to_numeric(temp_df["年初至今涨跌幅"], errors="coerce")
    df = temp_df
    # return df
    # print(df['代码'], df['名称'])
    # i = 0
    # for cc, nn in zip(df['代码'],df['名称']):
    #     i += 1
    #     print(i, cc, nn)
    #     if cc in code_list:
    #         print(i, cc, nn, code_list)
    df1 = df[df['代码'].isin(code_list)]
    return df1


def get_security_minute_data(n=0, stock='600031', start=0, count=800):
    '''
    获取分钟数据
    n数据类型
    0 5分钟K线
    1 15分钟K线
    2 30分钟K线
    3 1小时K线
    4 日K线
    7 1分钟
    8 1分钟K线
    marker市场0深圳1上海
    stock证券代码
    start开始位置
    count返回的数据长度
    '''
    i = 0
    while i != 1:
        try:
            stock = str(stock)[:6]
            marker = self.marker_type(stock=stock)
            df = self.api.get_security_bars(category=n, market=marker, code=stock, start=start, count=count)
            result = self.api.to_df(df)
            i = 1
            return result
        except:
            self.next_connect()

def get_security_week_data(self, stock='600031', start=0, count=100):
    '''
    获取股票周线数据
    stock证券代码
    count返回长度
    '''
    i = 0
    while i != 1:
        try:
            stock = str(stock)[:6]
            n = self.marker_type(stock=stock)
            df = self.api.get_security_bars(5, n, stock, start, count)
            result = self.api.to_df(df)
            i = 1
            return result
        except:
            self.next_connect()

def get_security_moth_data(self, stock='600031', start=0, count=100):
    '''
    获取股票月线数据
    stock证券代码
    count返回长度
    '''
    i = 0
    while i != 1:
        try:
            stock = str(stock)[:6]
            n = self.marker_type(stock=stock)
            df = self.api.get_security_bars(6, n, stock, start, count)
            result = self.api.to_df(df)
            i = 1
            return result
        except:
            self.next_connect()

def get_security_daily_data(self, stock='600031', start=0, count=100):
    '''
    获取股票日线数据
    stock证券代码
    count返回长度
    '''
    i = 0
    while i != 1:
        try:
            stock = str(stock)[:6]
            n = self.marker_type(stock=stock)
            df = self.api.get_security_bars(9, n, stock, start, count)
            result = self.api.to_df(df)
            i = 1
            return result
        except:
            self.next_connect()


def rename_stock_type_1(stock='600031'):
    '''
    将股票类型格式化
    stock证券代码
    1上海
    0深圳
    '''
    # if stock[:3] in ['600', '601', '603', '688', '510', '511',
    #                  '512', '513', '515', '113', '110', '118', '501'] or stock[:2] in ['11']:
    #     marker = 1
    # else:
    #     marker = 0
    # return marker, stock
    code = str(stock).zfill(6)
    if code.startswith('6'):
        return '1', code  # 沪市
    elif code.startswith('0') or code.startswith('3'):
        return '0', code  # 深市
    else:
        raise ValueError(f"无法识别的股票代码格式: {stock}")

def rename_stock_type(stock='600031'):
    '''
    将股票类型格式化
    stock证券代码
    1上海
    0深圳
    '''
    if stock[:3] in ['600', '601', '603', '688', '510', '511',
                     '512', '513', '515', '113', '110', '118', '501'] or stock[:2] in ['11']:
        marker = 1
    else:
        marker = 0
    result = [(marker, stock)]
    return result

def marker_type(stock='600031'):
    '''
    判断市场类型
    '''
    if stock[:3] in ['600', '601', '603', '688', '510', '511',
                     '512', '513', '515', '113', '110', '118', '501'] or stock[:2] in ['11']:
        marker = 1
    else:
        marker = 0
    return marker


def get_security_quotes_more(self, stock_list=['600031', '000001']):
    '''
    同时获取多只股票行情数据
    code_list股票列表
    [('market', 0),
          ('code', '000001'),
          ('active1', 2864),
          ('price', 9.19),
          ('last_close', 9.25),
          ('open', 9.23),
          ('high', 9.27),
          ('low', 9.16),
          ('reversed_bytes0', bytearray(b'\xbd\xc9\xec\x0c')),
          ('reversed_bytes1', -919),
          ('vol', 428899),
          ('cur_vol', 30),
          ('amount', 395218880.0),
          ('s_vol', 284703),
          ('b_vol', 144196),
          ('reversed_bytes2', 1),
          ('reversed_bytes3', 698),
          ('bid1', 9.18),
          ('ask1', 9.19),
          ('bid_vol1', 1078),
          ('ask_vol1', 5236),
          ('bid2', 9.17),
          ('ask2', 9.2),
          ('bid_vol2', 8591),
          ('ask_vol2', 3027),
          ('bid3', 9.16),
          ('ask3', 9.21),
          ('bid_vol3', 12638),
          ('ask_vol3', 3557),
          ('bid4', 9.15),
          ('ask4', 9.22),
          ('bid_vol4', 13234),
          ('ask_vol4', 2615),
          ('bid5', 9.14),
          ('ask5', 9.23),
          ('bid_vol5', 5377),
          ('ask_vol5', 6033),
          ('reversed_bytes4', 5768),
          ('reversed_bytes5', 1),
          ('reversed_bytes6', 16),
          ('reversed_bytes7', 83),
          ('reversed_bytes8', 20),
          ('reversed_bytes9', 0),
          ('active2', 2864)])]
    '''
    code_list = stock_list
    stock_list = []
    for i in code_list:
        stock = self.rename_stock_type(i)
        stock_list.append(stock[0])
    df = self.api.get_security_quotes(all_stock=stock_list)
    result = self.api.to_df(df)
    return result

def get_stock_info(stock_code: str) -> dict:
    """
    最终验证版本：修正行业ID和时间戳问题
    """
    # 验证后的字段映射表（2024年7月最新）
    FIELD_MAPPING = {
        "f57": "code",  # 股票代码
        "f58": "name",  # 股票名称
        "f116": "total_mv",  # 总市值（元）
        "f117": "circ_mv",  # 流通市值（元）
        "f198": "industry_id",  # 行业代码（已验证有效字段）
        "f127": "industry",  # 行业名称
        "f162": "pe_ttm",  # 动态市盈率（TTM）
        "f167": "pb",  # 市净率
        "f297": "update_time"  # 时间戳字段（需处理空值）
    }

    result = {
        "code": stock_code,
        "error": None
    }

    try:
        # === 请求配置 ===
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "http://quote.eastmoney.com/"
        }

        # === 市场代码检测 ===
        def get_market(code):
            if code.startswith(('6', '9', '5')): return '1'  # 沪市
            if code.startswith(('0', '3', '2')): return '0'  # 深市
            raise ValueError("不支持的股票代码")

        market = get_market(stock_code)
        secid = f"{market}.{stock_code}"

        # === 发送请求 ===
        url = "http://push2.eastmoney.com/api/qt/stock/get"
        params = {
            "ut": "fa5fd1943c7b386f172d6893dbfba10b",
            "fltt": "2",
            "invt": "2",
            "fields": ",".join(FIELD_MAPPING.keys()),
            "secid": secid,
            "cb": f"jsonp_{int(time.time() * 1000)}"
        }

        response = requests.get(url, headers=headers, params=params, timeout=(5, 5))
        response.raise_for_status()

        # === 数据解析 ===
        json_str = response.text.split('(', 1)[1].rsplit(')', 1)[0]
        raw_data = json.loads(json_str).get("data", {})

        # === 调试输出 ===
        # print("[调试] 接口原始数据:", json.dumps(raw_data, ensure_ascii=False, indent=2))  # 查看实际字段值

        # === 字段处理 ===
        for field, key in FIELD_MAPPING.items():
            value = raw_data.get(field)

            # 处理行业ID（兼容浮点格式）
            if key == "industry_id":
                result[key] = str(value).split(".")[0] if value else "N/A"

            # 处理时间戳（解决空值问题）
            elif key == "update_time":
                result[key] = value
                # try:
                #     # ts = int(value) // 1000 if value else 0
                #     ts = int(value)
                #     result[key] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts > 0 else "N/A"
                # except:
                #     result[key] = "N/A"

            # 处理市值（单位转换）
            elif key in ["total_mv", "circ_mv"]:
                result[key] = round(float(value or 0) / 1e8, 2)  # 转为亿元

            # 处理数值字段
            elif key in ["pe_ttm", "pb"]:
                result[key] = round(float(value or 0), 4)

            # 常规字段处理
            else:
                result[key] = str(value).strip() if value else ""

        # === 行业名称清洗 ===
        result["industry"] = result["industry"].split(";;")[0].replace("--", "-")

    except requests.RequestException as e:
        result["error"] = f"请求失败: {str(e)}"
    except json.JSONDecodeError:
        result["error"] = "接口响应数据格式异常"
    except Exception as e:
        result["error"] = f"处理错误: {str(e)}"

    return result

def get_stock_hist_data(stock='600031',start_date='20210101',end_date='20500101',data_type='D',count=8000):
    '''
    获取股票数据
    start_date=''默认上市时间
    - ``1`` : 分钟
        - ``5`` : 5 分钟
        - ``15`` : 15 分钟
        - ``30`` : 30 分钟
        - ``60`` : 60 分钟
        - ``101`` : 日
        - ``102`` : 周
        - ``103`` : 月
    fq=0股票除权
    fq=1前复权
    fq=2后复权
    '''
    #qmt数据
    qmt_dict={'1': '1m', '5': '5m', '15': '15m', '30m': '30m', '60': '60m', 'D': '1d', 'W': '1w', 'M': '1mon',
            "tick":'tick','1q':"1q","1hy":"1hy","1y":"1y"}
    data_dict = {'1': '1', '5': '5', '15': '15', '30': '30', '60': '60', 'D': '101', 'W': '102', 'M': '103'}
    if self.data_api=='qmt':
        try:
            stock_1=self.adjust_stock(stock=stock)
            period=qmt_dict.get(data_type,'D')
            xtdata.subscribe_quote(stock_code=stock_1,period=period,start_time=start_date,end_time=end_date,count=-1)
            data=xtdata.get_market_data_ex(stock_list=[stock_1],period=period,start_time=start_date,end_time=end_date,count=-1,dividend_type="front")
            data=data[stock_1]
            if data.shape[0]>0:
                #['date', 'open', 'close', 'high', 'low', 'volume', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
                data['date']=data.index.tolist()
                data['成交额']=data['amount']
                data['涨跌幅']=data['close'].pct_change()*100
                data['涨跌额']=data['close']-data['close'].shift(1)
                return data
            else:
                try:
                    stock=str(stock)[:6]
                    klt=data_dict[data_type]
                    klt=data_dict[data_type]
                    secid='{}.{}'.format(0,stock)
                    url = 'http://push2his.eastmoney.com/api/qt/stock/kline/get?'
                    params = {
                        'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
                        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
                        'beg': start_date,
                        'end': end_date,
                        'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
                        'rtntype':end_date,
                        'secid': secid,
                        'klt':klt,
                        'fqt': '1',
                        'cb': 'jsonp1668432946680'
                    }
                    res = requests.get(url=url, params=params)
                    text = res.text[19:len(res.text) - 2]
                    json_text = json.loads(text)
                    df = pd.DataFrame(json_text['data']['klines'])
                    df.columns = ['数据']
                    data_list = []
                    for i in df['数据']:
                        data_list.append(i.split(','))
                    data = pd.DataFrame(data_list)
                    columns = ['date', 'open', 'close', 'high', 'low', 'volume', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
                    data.columns = columns
                    for m in columns[1:]:
                        data[m] = pd.to_numeric(data[m])
                    data.sort_index(ascending=True,ignore_index=True,inplace=True)
                    return data
                except:
                    stock=str(stock)[:6]
                    data_dict = {'1': '1', '5': '5', '15': '15', '30': '30', '60': '60', 'D': '101', 'W': '102', 'M': '103'}
                    klt=data_dict[data_type]
                    klt=data_dict[data_type]
                    secid='{}.{}'.format(1,stock)
                    url = 'http://push2his.eastmoney.com/api/qt/stock/kline/get?'
                    params = {
                            'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
                            'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
                            'beg': start_date,
                            'end': end_date,
                            'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
                            'rtntype':end_date,
                            'secid': secid,
                            'klt':klt,
                            'fqt': '1',
                            'cb': 'jsonp1668432946680'
                    }
                    res = requests.get(url=url, params=params)
                    text = res.text[19:len(res.text) - 2]
                    json_text = json.loads(text)
                    df = pd.DataFrame(json_text['data']['klines'])
                    df.columns = ['数据']
                    data_list = []
                    for i in df['数据']:
                        data_list.append(i.split(','))
                    data = pd.DataFrame(data_list)
                    columns = ['date', 'open', 'close', 'high', 'low', 'volume', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
                    data.columns = columns
                    for m in columns[1:]:
                        data[m] = pd.to_numeric(data[m])
                    data.sort_index(ascending=True,ignore_index=True,inplace=True)
                    return data
        except Exception as e:
            print(e,stock_1,'qmt获取股票数据有问题切换到东方财富')
            try:
                stock=str(stock)[:6]
                klt=data_dict[data_type]
                klt=data_dict[data_type]
                secid='{}.{}'.format(0,stock)
                url = 'http://push2his.eastmoney.com/api/qt/stock/kline/get?'
                params = {
                    'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
                    'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
                    'beg': start_date,
                    'end': end_date,
                    'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
                    'rtntype':end_date,
                    'secid': secid,
                    'klt':klt,
                    'fqt': '1',
                    'cb': 'jsonp1668432946680'
                }
                res = requests.get(url=url, params=params)
                text = res.text[19:len(res.text) - 2]
                json_text = json.loads(text)
                df = pd.DataFrame(json_text['data']['klines'])
                df.columns = ['数据']
                data_list = []
                for i in df['数据']:
                    data_list.append(i.split(','))
                data = pd.DataFrame(data_list)
                columns = ['date', 'open', 'close', 'high', 'low', 'volume', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
                data.columns = columns
                for m in columns[1:]:
                    data[m] = pd.to_numeric(data[m])
                data.sort_index(ascending=True,ignore_index=True,inplace=True)
                return data
            except:
                stock=str(stock)[:6]
                data_dict = {'1': '1', '5': '5', '15': '15', '30': '30', '60': '60', 'D': '101', 'W': '102', 'M': '103'}
                klt=data_dict[data_type]
                klt=data_dict[data_type]
                secid='{}.{}'.format(1,stock)
                url = 'http://push2his.eastmoney.com/api/qt/stock/kline/get?'
                params = {
                        'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
                        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
                        'beg': start_date,
                        'end': end_date,
                        'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
                        'rtntype':end_date,
                        'secid': secid,
                        'klt':klt,
                        'fqt': '1',
                        'cb': 'jsonp1668432946680'
                }
                res = requests.get(url=url, params=params)
                text = res.text[19:len(res.text) - 2]
                json_text = json.loads(text)
                df = pd.DataFrame(json_text['data']['klines'])
                df.columns = ['数据']
                data_list = []
                for i in df['数据']:
                    data_list.append(i.split(','))
                data = pd.DataFrame(data_list)
                columns = ['date', 'open', 'close', 'high', 'low', 'volume', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
                data.columns = columns
                for m in columns[1:]:
                    data[m] = pd.to_numeric(data[m])
                data.sort_index(ascending=True,ignore_index=True,inplace=True)
                return data
    else:
        try:
            stock=str(stock)[:6]
            klt=data_dict[data_type]
            klt=data_dict[data_type]
            secid='{}.{}'.format(0,stock)
            url = 'http://push2his.eastmoney.com/api/qt/stock/kline/get?'
            params = {
                    'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
                    'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
                    'beg': start_date,
                    'end': end_date,
                    'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
                    'rtntype':end_date,
                    'secid': secid,
                    'klt':klt,
                    'fqt': '1',
                    'cb': 'jsonp1668432946680'
                }
            res = requests.get(url=url, params=params)
            text = res.text[19:len(res.text) - 2]
            json_text = json.loads(text)
            df = pd.DataFrame(json_text['data']['klines'])
            df.columns = ['数据']
            data_list = []
            for i in df['数据']:
                data_list.append(i.split(','))
            data = pd.DataFrame(data_list)
            columns = ['date', 'open', 'close', 'high', 'low', 'volume', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
            data.columns = columns
            for m in columns[1:]:
                data[m] = pd.to_numeric(data[m])
            data.sort_index(ascending=True,ignore_index=True,inplace=True)
            return data
        except:
            try:
                stock=str(stock)[:6]
                data_dict = {'1': '1', '5': '5', '15': '15', '30': '30', '60': '60', 'D': '101', 'W': '102', 'M': '103'}
                klt=data_dict[data_type]
                klt=data_dict[data_type]
                secid='{}.{}'.format(1,stock)
                url = 'http://push2his.eastmoney.com/api/qt/stock/kline/get?'
                params = {
                            'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
                            'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
                            'beg': start_date,
                            'end': end_date,
                            'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
                            'rtntype':end_date,
                            'secid': secid,
                            'klt':klt,
                            'fqt': '1',
                            'cb': 'jsonp1668432946680'
                    }
                res = requests.get(url=url, params=params)
                text = res.text[19:len(res.text) - 2]
                json_text = json.loads(text)
                df = pd.DataFrame(json_text['data']['klines'])
                df.columns = ['数据']
                data_list = []
                for i in df['数据']:
                    data_list.append(i.split(','))
                data = pd.DataFrame(data_list)
                columns = ['date', 'open', 'close', 'high', 'low', 'volume', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
                data.columns = columns
                for m in columns[1:]:
                    data[m] = pd.to_numeric(data[m])
                data.sort_index(ascending=True,ignore_index=True,inplace=True)
                return data
            except Exception as e:
                print(e,stock,'东方财富数据获取有问题切换到qmt')
                stock_1=self.adjust_stock(stock=stock)
                period=qmt_dict.get(data_type,'D')
                xtdata.subscribe_quote(stock_code=stock_1,period=period,start_time=start_date,end_time=end_date,count=-1)
                data=xtdata.get_market_data_ex(stock_list=[stock_1],period=period,start_time=start_date,end_time=end_date,count=-1,dividend_type="front")
                data=data[stock_1]
                #['date', 'open', 'close', 'high', 'low', 'volume', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
                data['date']=data.index.tolist()
                data['成交额']=data['amount']
                data['涨跌幅']=data['close'].pct_change()*100
                data['涨跌额']=data['close']-data['close'].shift(1)
                return data


def get_stock_hist_minutes_data_em(stock='600031', data_type='5', count=8000):
    """增强版股票分钟线获取（解决模拟数据问题）"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Referer': 'http://push2his.eastmoney.com/'
    }

    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    try:
        # 股票代码转换逻辑增强
        def rename_stock_code(code):
            if code.startswith(('6', '9', '688')):  # 上证股票（含科创板）
                return '1', code
            elif code.startswith(('0', '3')):  # 深证股票
                return '0', code
            else:
                raise ValueError(f"无效股票代码: {code}")

        market_code, stock_code = rename_stock_code(stock)
        secid = f"{market_code}.{stock_code}"

        # 构造请求参数
        url = 'http://push2his.eastmoney.com/api/qt/stock/kline/get'
        params = {
            'fields1': 'f1,f2,f3,f4,f5',
            'fields2': 'f51,f52,f53,f54,f55,f56,f57',
            'secid': secid,
            'klt': data_type,
            'fqt': '1',
            'lmt': 100000,
            'end': '20500101',  # 固定获取最新数据
        }
        print(params)

        res = session.get(url, params=params, headers=headers, timeout=30)
        res.raise_for_status()
        # 解析JSONP或直接JSON
        raw_text = res.text
        print(raw_text)

        if '(' in raw_text and ')' in raw_text:
            json_str = re.findall(r'\((.*?)\)', raw_text)[0]
            data = json.loads(json_str)
        else:
            data = json.loads(raw_text)

        # 检查是否为模拟数据
        if 'data' not in data or not data['data'].get('klines'):
            print(f"警告：股票 {stock} 无历史数据或返回模拟数据")
            return pd.DataFrame()

        klines = data['data']['klines']
        df = pd.DataFrame([k.split(',') for k in klines],
                          columns=['时间', '开盘', '收盘', '最高', '最低', '成交量', '成交额'])

        # 过滤未来日期数据（模拟数据特征）
        df['时间'] = pd.to_datetime(df['时间'])
        current_time = pd.Timestamp.now()
        df = df[df['时间'] <= current_time]  # 移除未来时间戳

        if df.empty:
            print(f"警告：股票 {stock} 数据均为模拟数据")
            return pd.DataFrame()

        # 类型转换
        df[['开盘', '收盘', '最高', '最低']] = df[['开盘', '收盘', '最高', '最低']].astype(float)
        df[['成交量', '成交额']] = df[['成交量', '成交额']].astype(int)

        return df.sort_values('时间')

    except Exception as e:
        print(f"请求失败: {str(e)}")
        return pd.DataFrame()


def get_industry_mapping():
    """ 动态获取行业代码映射 """
    url = "http://push2.eastmoney.com/api/qt/club/get"
    params = {
        "pn": 1,
        "pz": 1000,
        "fields": "f14,f12,f3",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
    }
    res = requests.get(url, params=params)
    data = res.json()["data"]["diff"]
    return {item["f12"]: item["f14"] for item in data}

if __name__ == "__main__":
    # stock_code_list = ["600570.SH", "002261.SZ"]
    # for stock_code in stock_code_list:
    #     lastest_price = get_latest_price(stock_code, is_download=True)
    #     print('stock_code=', stock_code, ';lastest_price = ', lastest_price)
    #     last_1d_close_price = get_yesterday_close_price(stock_code)
    #     last_1d_c_price = get_close_price(stock_code)
    #     print('last_1d_close_price=', last_1d_close_price, last_1d_c_price)
    #
    #     last_2d_c_price = get_close_price(stock_code, last_n=2)
    #     print('last_2d_close_price=', last_2d_c_price)

    # a = get_stock_hist_data_em("600570")
    # print(a)

    # print(a.columns)
    # b = get_stock_now_data(code_list=['002736', '603237', '600865', '601866', '600824'])
    # print(b)
    # print(get_stock_info("600519"))  # 贵州茅台
    # print(get_stock_info("300750"))  # 宁德时代
    # print(get_stock_info("000001"))  # 平安银行
    # print(get_ut_token())
    # print(get_stock_hist_data_em_with_retry_v2())
    #print(get_stock_hist_data_em_with_retry())
    # print(get_stock_now_data())


    stfile = 'D:/tool/dataset/stock_pools.json'
    sc_dict = json.load(open(stfile))
    # subscribe_whole_list = list(set([info_dict.get("code") for code, info_dict in sc_dict.items()]))
    subscribe_whole_list = list(set([code for code, info_dict in sc_dict.items()]))
    start = time.time()
    df = get_batch_stock_data_em_from_qmt(stock_list=subscribe_whole_list, start_date='20210101', end_date='20500101',
                                         data_type='1',
                                         count=8000)
    end = time.time()
    print(f"get_batch_stock_data spend:{(end-start)/1000}")
    # print(df)
    all_num = 0
    for i, code in enumerate(subscribe_whole_list):
        info = df[code]

        print(f"ind={i}, code={code}, info={info}")
        if i > 60:
            break


