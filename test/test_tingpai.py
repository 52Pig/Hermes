from xtquant import xtdata


stock_code = "600520.SH"
instrument_detail = xtdata.get_instrument_detail(stock_code)
## 是否停牌
ins_status = instrument_detail.get("InstrumentStatus")
print(ins_status)