# coding:gbk
'''
���ڻ�ת�ز�ģ��ʾ������ʵ�̽��ײ��ԣ�

�������������뵱ǰ��Ʊ10000��
������60s������������MACD(12,26,9)��,����MACD>0��MACD_pre<0��ʱ������100��,MACD<0,MACD_pre>0��ʱ������100��
��ÿ�ղ����Ĺ�Ʊ��������ԭ�в�λ,��������ǰ�Ѳ�λ����������ǰ�Ĳ�λ
���������ڸ��ɷ�����������
'''
import numpy as np
import pandas as pd
import talib


def init(ContextInfo):
	MarketPosition = {}
	ContextInfo.MarketPosition = MarketPosition  # ��ʼ���ֲ�
	ContextInfo.set_universe([ContextInfo.stockcode + '.' + ContextInfo.market])
	ContextInfo.first = 0
	ContextInfo.Lots = 100  # �趨��������
	ContextInfo.day = [0, 0]
	ContextInfo.ending = 0
	ContextInfo.total = 10000
	ContextInfo.accountID = 'testS'


def handlebar(ContextInfo):
	d = ContextInfo.barpos
	if d < 35:
		return
	startdate = timetag_to_datetime(ContextInfo.get_bar_timetag(d - 35), '%Y%m%d%H%M%S')
	enddate = timetag_to_datetime(ContextInfo.get_bar_timetag(d), '%Y%m%d%H%M%S')
	##print startdate,enddate
	date = timetag_to_datetime(ContextInfo.get_bar_timetag(d), '%Y-%m-%d %H:%M:%S')
	print('����', date)
	flage = False
	singleemited = False

	df = ContextInfo.get_market_data(['close'], stock_code=ContextInfo.get_universe(), start_time=startdate,
									end_time=enddate, period=ContextInfo.period)
	# print df
	if df.empty:
		return
	# print df
	if ContextInfo.first == 0:
		order_shares(ContextInfo.get_universe()[0], ContextInfo.total, 'fix', df.iloc[-1, 0], ContextInfo,
						ContextInfo.accountID)
		flage = True
		singleemited = True
		ContextInfo.first = 1
		ContextInfo.day[-1] = date[8:10]
		ContextInfo.turnaround = [0, 0]
		ContextInfo.MarketPosition[ContextInfo.get_universe()[0]] = ContextInfo.total
		return

	ContextInfo.day[0] = date

	# 14:55:00֮���ٽ���
	if int(date[-8:-6] + date[-5:-3]) > 1455:
		return
	avaliable = get_avaliable(ContextInfo.accountID, 'STOCK')

	holding = get_holdings(ContextInfo.accountID, 'STOCK')
	if ContextInfo.get_universe()[0] not in list(holding.keys()):
		holding[ContextInfo.get_universe()[0]] = 0

	# ����MACD��
	if ContextInfo.total >= 0:
		recent_date = np.array(df.iloc[-35:, 0])

		macd = talib.MACD(recent_date)[0][-1]
		macd_pre = talib.MACD(recent_date)[0][-2]

		# ����MACD>0�򿪲�,С��0��ƽ��

		if date[-8:-3] != '14:55':
			if macd > 0 and macd_pre < 0:
				# ����MACD>0�򿪲�,С��0��ƽ��
				if avaliable > df.iloc[-1, 0] * ContextInfo.Lots * 100:
					order_shares(ContextInfo.get_universe()[0], ContextInfo.Lots, 'fix', df.iloc[-1, 0], ContextInfo,
									ContextInfo.accountID)
					flage = True
					singleemited = True
					ContextInfo.MarketPosition[ContextInfo.get_universe()[0]] += ContextInfo.Lots
					print (ContextInfo.get_universe()[0], 'open position at market price', ContextInfo.Lots, '��')

			elif macd < 0 and macd_pre > 0 and holding[ContextInfo.get_universe()[0]] >= ContextInfo.Lots:
				order_shares(ContextInfo.get_universe()[0], -ContextInfo.Lots, 'fix', df.iloc[-1, 0], ContextInfo,
								ContextInfo.accountID)
				flage = False
				singleemited = True
				print(ContextInfo.get_universe()[0], 'close position at market price', ContextInfo.Lots, '��')
				ContextInfo.MarketPosition[ContextInfo.get_universe()[0]] -= ContextInfo.Lots
			# �ٽ�����ʱ����λ��������������ת���в�λ

		else:
			if ContextInfo.MarketPosition[ContextInfo.get_universe()[0]] > ContextInfo.total:
				order_shares(ContextInfo.get_universe()[0],
								-(ContextInfo.MarketPosition[ContextInfo.get_universe()[0]] - ContextInfo.total), 'fix',
								df.iloc[-1, 0], ContextInfo, ContextInfo.accountID)
				flage = False
				singleemited = True
				# print ContextInfo.get_universe()[0], '��ת�����м۵�ƽ���', ContextInfo.MarketPosition[ContextInfo.get_universe()[0]] - ContextInfo.total, '��'
				ContextInfo.MarketPosition[ContextInfo.get_universe()[0]] = ContextInfo.total

			if ContextInfo.MarketPosition[ContextInfo.get_universe()[0]] < ContextInfo.total:
				order_shares(ContextInfo.get_universe()[0],
								(ContextInfo.total - ContextInfo.MarketPosition[ContextInfo.get_universe()[0]]), 'fix',
								df.iloc[-1, 0], ContextInfo, ContextInfo.accountID)
				flage = True
				singleemited = True
				# print ContextInfo.get_universe()[0], '��ת�����м۵������', ContextInfo.total - ContextInfo.MarketPosition[ContextInfo.get_universe()[0]] , '��'
				ContextInfo.MarketPosition[ContextInfo.get_universe()[0]] = ContextInfo.total

		# ���¹�ȥ����������
		'''ContextInfo.day[-1] = ContextInfo.day[0]
		if singleemited:
			if flage:
				ContextInfo.paint('do_buy',1,-1,0,"yellow",'noaxis')
				ContextInfo.paint('do_sell',0,-1,0,"red",'noaxis')
			else:
				ContextInfo.paint('do_buy',0,-1,0,"yellow",'noaxis')
				ContextInfo.paint('do_sell',1,-1,0,"red",'noaxis')'''


# ContextInfo.paint('holding',ContextInfo.MarketPosition[ContextInfo.get_universe()[0]],-1,0)

def get_avaliable(accountid, datatype):
	result = 0
	resultlist = get_trade_detail_data(accountid, datatype, "ACCOUNT")
	for obj in resultlist:
		result = obj.m_dAvailable
	return result


def get_holdings(accountid, datatype):
	holdinglist = {}
	resultlist = get_trade_detail_data(accountid, datatype, "POSITION")
	for obj in resultlist:
		holdinglist[obj.m_strInstrumentID + "." + obj.m_strExchangeID] = obj.m_nCanUseVolume
	return holdinglist