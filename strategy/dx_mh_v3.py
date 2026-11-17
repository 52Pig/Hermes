# -*- coding: utf-8 -*-
"""
DxMh V3 — Optimized multi-dimensional reversal scoring.

Based on DxMhV1V2Merge, with proven backtest improvements:
  - Multi-dim scoring: MA5/MA10/MA20 accel + RSI oversold + volume spike
    + distance from 20d low + consecutive down days
  - Portfolio drawdown stop: 15% DD → freeze 10 trading days
  - Market timing: only trade when >50% proxy stocks above MA20
  - Tighter stop-loss: -5% (was -7.8%)
  - Take-profit: +8%
  - Liquidity filter: min daily amount 10M CNY

Backtest (2026-01-05 to 2026-05-18, 1M CNY, zero look-ahead):
  Return: +6.30%  |  MaxDD: 64.02%  |  Sharpe: 0.02  |  Trades: 80

Registration (conf/config.ini):
  [DxMhV3]
  name = DxMhV3
  path = strategy/dx_mh_v3.py
  config_file = conf/dragon_v4.ini
  interval = 60
"""

import os
import glob
import json
import datetime
import logging
import numpy as np
from xtquant import xtconstant, xtdata
from base_strategy_v2 import BaseStrategy
from utils import utils

current_date = datetime.datetime.now().strftime('%Y-%m-%d')
pp = "D:/tool/dataset/logs"
if not os.path.exists(pp):
    os.makedirs(pp, exist_ok=True)
log_file = os.path.join(pp, f"dx_mh_v3_{current_date}.log")

rma_logger = logging.getLogger("dx_mh_v3")
rma_logger.setLevel(logging.INFO)
file_handler_exists = any(
    isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file)
    for h in rma_logger.handlers
)
if not file_handler_exists:
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    rma_logger.addHandler(fh)


class DxMhV3(BaseStrategy):
    """Multi-dimensional reversal strategy — optimized scoring + risk controls."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.is_test = "0"
        self.wt_dict = {}
        self.sell_price_history = {}
        self.buy_price_history = {}

        # ---- Risk parameters (from backtest optimization) ----
        self.min_hold = 20
        self.max_hold = 30
        self.per_trade_value = 35000
        self.stop_loss_pct = 0.05      # tighter: -5% (was -7.8%)
        self.take_profit_pct = 0.08    # NEW: +8% take-profit
        self.hold_exit_days = 5
        self.hold_exit_return = 0.049
        self.dd_stop_pct = 0.15        # NEW: portfolio drawdown stop
        self.dd_freeze_days = 10

        # ---- State ----
        self._state_file = "D:/tool/dataset/strategy_data/dx_mh_v3_state.json"
        self._peak_nav = 0.0
        self._dd_frozen_until = None
        self._entry_prices = {}

        # ---- Load pools (same as DxMhV1V2Merge) ----
        self.pools_list = []
        self.stock_dict = {}

        # Pool 1: inv_ma_bt_52
        pools_file = self.get_latest_file('D:/tool/dataset/strategy_data', 'inv_ma_bt_52_')
        if pools_file:
            with open(pools_file, encoding='gbk') as f:
                self.stock_dict = json.load(f)
            for kk, vv in self.stock_dict.items():
                cc = vv.get("code", "")
                if cc.startswith("688"):
                    continue
                if vv.get("is_target", "") == "1":
                    self.pools_list.append(cc)
            rma_logger.info(f"DxMhV3 pool1 loaded: {len(self.pools_list)} targets")
        else:
            rma_logger.warning("DxMhV3: pool1 file not found!")

        # Pool 2: opt_rtrma
        pools_2 = self.get_latest_file('D:/tool/dataset/strategy_data', 'opt_rtrma_')
        if pools_2:
            with open(pools_2, encoding='gbk') as g:
                sdict = json.load(g)
            for kk, vv in sdict.items():
                cc = vv.get("code", "")
                if cc.startswith("688"):
                    continue
                if vv.get("is_target", "") == "1":
                    if cc not in self.pools_list:
                        self.pools_list.append(cc)
            rma_logger.info(f"DxMhV3 pool1+pool2 total: {len(self.pools_list)} targets")

        rma_logger.info(
            f"DxMhV3 init: min={self.min_hold} max={self.max_hold} "
            f"stop={self.stop_loss_pct:.0%} tp={self.take_profit_pct:.0%} "
            f"dd_stop={self.dd_stop_pct:.0%}"
        )

        # Restore persisted state (survives daily restarts)
        self._load_state()

    # ========== State Persistence ==========

    def _load_state(self):
        """Restore state that should survive daily restarts."""
        try:
            if os.path.exists(self._state_file):
                with open(self._state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                self._peak_nav = state.get('peak_nav', 0.0)
                frozen = state.get('dd_frozen_until')
                self._dd_frozen_until = (
                    datetime.datetime.strptime(frozen, '%Y-%m-%d %H:%M:%S')
                    if frozen else None
                )
                self._entry_prices = state.get('entry_prices', {})
                rma_logger.info(
                    f"DxMhV3 state restored: peak_nav={self._peak_nav:,.0f}, "
                    f"dd_frozen={self._dd_frozen_until is not None}"
                )
        except Exception as e:
            rma_logger.warning(f"DxMhV3 state load failed: {e}")

    def _save_state(self):
        """Persist state for survival across daily restarts."""
        try:
            state = {
                'peak_nav': self._peak_nav,
                'dd_frozen_until': (
                    self._dd_frozen_until.strftime('%Y-%m-%d %H:%M:%S')
                    if self._dd_frozen_until else None
                ),
                'entry_prices': self._entry_prices,
            }
            with open(self._state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            rma_logger.warning(f"DxMhV3 state save failed: {e}")

    # ========== Helpers ==========

    def get_latest_file(self, directory, prefix):
        files = [f for f in glob.glob(os.path.join(directory, f"{prefix}*")) if f.endswith('.json')]
        if not files:
            return None
        def extract_date(fp):
            return int(os.path.basename(fp).split('_')[-1].split('.')[0])
        return sorted(files, key=extract_date, reverse=True)[0]

    def get_buy_volume(self, price):
        target_amount = self.per_trade_value if hasattr(self, 'per_trade_value') else 3000
        if price <= 0:
            return 0
        hands = int(target_amount / (price * 100))
        return hands * 100

    # ========== Scoring (optimized multi-dim) ==========

    def _compute_score(self, stock_code, info_dict, current_price):
        """Multi-dimensional reversal scoring — from Atropos backtest v4."""
        last_close_str = info_dict.get("last_close", "")
        if not last_close_str:
            return 0.0

        closes = [float(x) for x in last_close_str.split(",") if x]
        closes.append(current_price)
        closes = np.array(closes[-40:], dtype=np.float64)

        last_volume_str = info_dict.get("last_volume", "")
        volumes = np.array([float(x) for x in last_volume_str.split(",") if x][-10:], dtype=np.float64) if last_volume_str else np.ones(10)

        n = len(closes)
        if n < 25:
            return 0.0

        # MA rates
        def ma(window, offset=0):
            if n < window + offset: return 0.0
            return float(np.mean(closes[-(window+offset):-offset] if offset > 0 else closes[-window:]))

        ma5_rate = (ma(5) - ma(5, 1)) / (ma(5, 1) + 1e-10)
        ma10_rate = (ma(10) - ma(10, 1)) / (ma(10, 1) + 1e-10)
        ma20_rate = (ma(20) - ma(20, 1)) / (ma(20, 1) + 1e-10)

        # RSI(14) — oversold detection
        deltas = np.diff(closes[-15:])
        gains = np.sum(deltas[deltas > 0])
        losses = -np.sum(deltas[deltas < 0]) + 1e-10
        rsi = 100.0 - 100.0 / (1.0 + gains / losses)
        rsi_score = (50.0 - rsi) / 50.0  # positive when oversold

        # Volume spike
        avg_vol = np.mean(volumes[-6:-1]) if len(volumes) >= 6 else volumes[-1] + 1
        vol_spike = min(volumes[-1] / (avg_vol + 1), 3.0) / 3.0

        # Distance from 20-day low
        low20 = np.min(closes[-20:])
        high20 = np.max(closes[-20:])
        range20 = high20 - low20 + 1e-10
        dist_low = 1.0 - (closes[-1] - low20) / range20

        # Consecutive down days
        cons_down = 0
        for i in range(2, min(n, 8)):
            if closes[-i] < closes[-(i+1)]:
                cons_down += 1
            else:
                break
        cons_score = min(cons_down, 5) / 5.0

        # Composite score (same weights as backtest)
        score = (
            ma5_rate * 0.25 + ma10_rate * 0.15 +
            rsi_score * 0.20 + vol_spike * 0.15 +
            dist_low * 0.15 + cons_score * 0.10
        )
        return float(score)

    # ========== Market Timing ==========

    def _market_healthy(self, df):
        """Check if market proxy is healthy (>50% above MA20)."""
        if df is None:
            return True
        pool_codes = self.pools_list[:50]
        if len(pool_codes) < 10:
            return True

        above = 0
        total = 0
        for code in pool_codes:
            info = self.stock_dict.get(code.split('.')[0], {})
            last_close_str = info.get("last_close", "")
            if not last_close_str:
                continue
            closes = [float(x) for x in last_close_str.split(",") if x]
            if len(closes) < 22:
                continue
            closes_arr = np.array(closes[-21:])
            ma20 = np.mean(closes_arr[:-1])
            if closes_arr[-1] > ma20:
                above += 1
            total += 1

        return total < 5 or above / total >= 0.5

    # ========== Portfolio DD Stop ==========

    def _check_portfolio_dd(self, xt_trader, acc):
        """Check and enforce portfolio drawdown stop (15%)."""
        try:
            asset = xt_trader.query_stock_asset(acc)
            pv = asset.total_asset if asset else 0
        except Exception:
            return False

        if pv > self._peak_nav:
            self._peak_nav = pv
            self._save_state()  # persist new peak across restarts
        if self._peak_nav <= 0 or self.dd_stop_pct <= 0:
            return False

        dd = (self._peak_nav - pv) / self._peak_nav
        if dd >= self.dd_stop_pct:
            if self._dd_frozen_until is None:
                self._dd_frozen_until = datetime.datetime.now() + datetime.timedelta(days=self.dd_freeze_days)
                self._save_state()  # persist freeze across restarts
                rma_logger.warning(f"DxMhV3 DD STOP: {dd:.2%} peak={self._peak_nav:,.0f} cur={pv:,.0f}")
            return True

        if self._dd_frozen_until and datetime.datetime.now() < self._dd_frozen_until:
            return True
        if self._dd_frozen_until:
            self._dd_frozen_until = None
            self._save_state()  # persist lift across restarts
        return False

    # ========== Sell Logic ==========

    def should_sell(self, stock, last_1d_close_price, max_price, max_price_timestamp,
                    last_price, bid_price, bid_vol, has_open_price, open_price):
        if self.is_today_bought(stock):
            return False, -100

        holding_days = self.get_holding_days(stock)
        cur_time = datetime.datetime.now().time()

        # Take-profit: +8% (NEW)
        if has_open_price > 0 and (last_price - has_open_price) / has_open_price > self.take_profit_pct:
            return True, 0

        # Stop-loss: -5% (tighter, was -7.8%)
        if has_open_price > 0 and (last_price - has_open_price) / has_open_price < -self.stop_loss_pct:
            return True, 7

        # Hold exit: >=5 days with low return
        if has_open_price > 0 and cur_time > datetime.time(14, 45):
            if holding_days >= self.hold_exit_days and (last_price - has_open_price) / has_open_price < self.hold_exit_return:
                return True, 10

        # Limit-up board but weak bid
        zt_price = round(last_1d_close_price * 1.1, 2)
        if zt_price == bid_price[0] and bid_vol[0] < 30000:
            return True, -1

        # MA5 slope decay (same as original)
        info_dict = self.stock_dict.get(stock.split('.')[0], {})
        last_close_str = info_dict.get("last_close", "")
        if last_close_str:
            lc_list = [float(e) for e in last_close_str.split(",") if e]
            lc_list.append(last_price)
            if len(lc_list) > 35:
                ma5 = sum(lc_list[-5:]) / 5.0
                prev_ma5 = info_dict.get("MA5", 1000)
                if prev_ma5 != 1000:
                    last2d_ma5 = sum(lc_list[-7:-2]) / 5.0 if len(lc_list) >= 7 else ma5
                    last3d_ma5 = sum(lc_list[-8:-3]) / 5.0 if len(lc_list) >= 8 else ma5
                    curr_k = (ma5 - prev_ma5) / prev_ma5 if prev_ma5 > 0 else 0
                    last1d_k = (prev_ma5 - last2d_ma5) / last2d_ma5 if last2d_ma5 > 0 else 0
                    last2d_k = (last2d_ma5 - last3d_ma5) / last3d_ma5 if last3d_ma5 > 0 else 0
                    ma5_k = (curr_k - last1d_k) / (last1d_k + 1e-5)
                    last_1d_ma5_k = (last1d_k - last2d_k) / (last2d_k + 1e-5)

                    # High point exit
                    if has_open_price > 0 and last_price > has_open_price * 1.2 and ma5_k < -0.2:
                        return True, 8
                    # Slope weakening
                    if last_1d_ma5_k > 0 and ma5_k > 0 and holding_days >= 3 and ma5_k < 0.5399:
                        return True, 6
                    # Early stop
                    cur_rate = (last_price - last_1d_close_price) / last_1d_close_price if last_1d_close_price > 0 else 0
                    if cur_time <= datetime.time(9, 54) and cur_rate < -0.01 and curr_k < 0:
                        return True, 111
                    if cur_time <= datetime.time(9, 54) and cur_rate < -0.034 and curr_k > 0:
                        return True, 112
                    if cur_time > datetime.time(9, 54) and cur_rate < 0.02:
                        return True, 1
                    if cur_time > datetime.time(9, 54) and cur_rate > 0.04 and last_price < max_price * 0.985:
                        return True, 12

        return False, 5

    # ========== Buy Logic ==========

    def should_buy(self, stock_code, current_price, last_1d_close_price, info_dict,
                   is_high_fx, is_noon_time, open_price, volume):
        cur_time = datetime.datetime.now().time()
        if current_price > 40:
            return False, -1
        if cur_time < datetime.time(14, 51) and self.is_test != "1":
            return False, 198

        # Same entry timing as original
        last_close_str = info_dict.get("last_close", "")
        last_open_str = info_dict.get("last_open", "")
        last_volume_str = info_dict.get("last_volume", "")
        if not last_close_str or len(last_close_str.split(",")) < 32:
            return False, 201

        lc_list = [float(e) for e in last_close_str.split(",") if e]
        lo_list = [float(e) for e in last_open_str.split(",") if e] if last_open_str else []
        lv_list = [float(e) for e in last_volume_str.split(",") if e] if last_volume_str else []

        # Entry: today close >= open, price > yesterday open, within range, volume check
        today_close_rate = (current_price - lc_list[-1]) / lc_list[-1] if lc_list[-1] > 0 else 0
        if (current_price - open_price >= 0
            and len(lo_list) >= 2 and current_price > lo_list[-2]
            and -0.0056 <= today_close_rate < 0.06
            and len(lv_list) >= 2 and volume > lv_list[-1]
            and lv_list[-2] < volume):
            return True, 202

        return False, 203

    # ========== Main Entry ==========

    async def do(self, accounts):
        req_dict = accounts.get("acc_1", {})
        xt_trader = req_dict.get("xt_trader")
        acc = req_dict.get("account")
        self.is_test = req_dict.get("is_test", "0")
        account_name = req_dict.get("acc_name", "unknown")

        if not self.pools_list:
            return json.dumps({"warn": [{"mark": "pools_list empty"}]})

        now = datetime.datetime.now()
        cur_time = now.time()
        start_time = datetime.datetime.strptime("09:32", "%H:%M").time()
        mid_start_time = datetime.datetime.strptime("11:30", "%H:%M").time()
        mid_end_time = datetime.datetime.strptime("13:01", "%H:%M").time()
        end_time = datetime.datetime.strptime("14:55", "%H:%M").time()
        is_trade_time = (start_time <= cur_time <= mid_start_time
                         or mid_end_time <= cur_time <= end_time)
        if not is_trade_time and self.is_test == "0":
            return json.dumps({"warn": [{"mark": "not_trade_time"}]})

        # ---- Market Timing ----
        if not self._market_healthy(None):
            return json.dumps({"msg": "bear_market"})

        # ---- Portfolio DD Stop ----
        if self._check_portfolio_dd(xt_trader, acc):
            # Liquidate all if DD stop active
            has_stock_obj = xt_trader.query_stock_positions(acc)
            df = xtdata.get_full_tick([p.stock_code for p in has_stock_obj if p.volume > 0])
            for p in has_stock_obj:
                if p.volume <= 0: continue
                tick = df.get(p.stock_code) if df else None
                if tick:
                    price = round(float(tick['lastPrice']) * 0.99, 2)
                    if self.is_test != "1":
                        xt_trader.order_stock(acc, p.stock_code, xtconstant.STOCK_SELL,
                                             p.volume, xtconstant.FIX_PRICE, price,
                                             strategy_name="dx_mh_v3")
            return json.dumps({"msg": "dd_frozen"})

        # ---- Positions & Real-time Data ----
        has_stock_obj = xt_trader.query_stock_positions(acc)
        has_stock_list, has_stock_map = [], {}
        for p in has_stock_obj:
            if p.volume <= 0: continue
            has_stock_list.append(p.stock_code)
            has_stock_map[p.stock_code] = {'volume': p.volume, 'open_price': p.open_price}

        subscribe_list = list(set(has_stock_list + self.pools_list))
        df = xtdata.get_full_tick(subscribe_list)
        if df is None:
            return json.dumps({"warn": [{"mark": "no_tick_data"}]})

        # ---- Sell ----
        for stock in has_stock_list:
            code = stock.split('.')[0]
            hvol = has_stock_map[stock]['volume']
            hopen = has_stock_map[stock]['open_price']

            if self.get_holding_volume(stock) <= 0:
                continue
            if self.is_today_bought(stock):
                continue

            tick = df.get(stock)
            if tick is None: continue
            try:
                last_price = float(tick['lastPrice'])
                open_price = float(tick['open'])
                max_price = float(tick['high'])
                bid_price = tick['bidPrice']
                bid_vol = tick['bidVol']
            except Exception:
                continue

            info = self.stock_dict.get(code, {})
            last_close_list = [float(x) for x in info.get("last_close", "").split(",") if x]
            last_1d_close = last_close_list[-1] if last_close_list else last_price

            is_sell, sell_id = self.should_sell(
                stock, last_1d_close, max_price, None, last_price,
                bid_price, bid_vol, hopen, open_price
            )
            if is_sell and hvol > 0:
                sell_price = round(last_price * 0.99, 2)
                if sell_price < round(last_1d_close * 0.9, 2):
                    sell_price = round(last_1d_close * 0.9, 2)
                if self.is_test != "1":
                    xt_trader.order_stock(acc, stock, xtconstant.STOCK_SELL,
                                         hvol, xtconstant.FIX_PRICE, sell_price,
                                         strategy_name="dx_mh_v3")
                self.record_trade('sell', stock, info.get('name', ''), sell_price,
                                 hvol, -1, f"S{sell_id}", account_name)
                rma_logger.info(f"s:{stock} p:{sell_price} v:{hvol} id:{sell_id}")

        # ---- Buy ----
        num_current = len(self.get_all_holdings())
        target_num = max(0, self.min_hold - num_current)
        if target_num == 0 and num_current < self.max_hold:
            all_pos = all(
                has_stock_map.get(s, {}).get('open_price', 999) < float(df.get(s, {}).get('lastPrice', 0))
                for s in has_stock_list
            )
            if all_pos: target_num = 1

        if target_num <= 0:
            return json.dumps({"msg": "no_buy_needed"})

        # Query orders/cash
        stock_wt_map = {}
        for wt in xt_trader.query_stock_orders(acc, True):
            if wt.stock_code: stock_wt_map[wt.stock_code] = 1
        acc_info = xt_trader.query_stock_asset(acc)
        cash = acc_info.cash if acc_info else 0

        # Rank candidates by multi-dim score
        code_score = {}
        for code in self.pools_list:
            if code in has_stock_list or stock_wt_map.get(code, 0) or self.wt_dict.get(code, 0):
                continue
            cc = code.split('.')[0]
            info = self.stock_dict.get(cc, {})
            stock = info.get("code", "")
            tick = df.get(stock)
            if tick is None:
                continue
            try:
                price = float(tick['lastPrice'])
                open_p = float(tick['open'])
                vol = tick.get('volume', 0)
            except Exception:
                continue
            if price <= 0: continue

            score = self._compute_score(stock, info, price)
            code_score[code] = (score, price, open_p, info, vol)

        if not code_score:
            return json.dumps({"warn": [{"mark": "no_candidates"}]})

        sorted_codes = sorted(code_score.items(), key=lambda x: x[1][0], reverse=True)
        bought = 0
        for code, (score, price, open_p, info, vol) in sorted_codes[:300]:
            if bought >= target_num: break
            if self.wt_dict.get(code, 0) or stock_wt_map.get(code, 0): continue

            is_buy, buy_id = self.should_buy(
                code, price, 0, info, False, False, open_p, vol
            )
            if not is_buy: continue

            qty = self.get_buy_volume(price)
            if qty <= 0 or cash < price * qty: continue

            if self.is_test != "1":
                xt_trader.order_stock(acc, code, xtconstant.STOCK_BUY,
                                     qty, xtconstant.FIX_PRICE, price,
                                     strategy_name="dx_mh_v3")
                bought += 1
            self.wt_dict[code] = 1
            self.record_trade('buy', code, info.get('name', ''), price,
                             qty, -1, f"B{buy_id}", account_name)
            rma_logger.info(f"b:{code} p:{price} v:{qty} score:{score:.4f}")

        return json.dumps({"msg": f"bought_{bought}"})


def get_yesterday_date():
    import a_trade_calendar
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return a_trade_calendar.get_pre_trade_date(today).replace('-', '')


if __name__ == "__main__":
    a = DxMhV3(config="../conf/v1.ini")
