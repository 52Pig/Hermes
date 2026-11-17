# -*- coding: utf-8 -*-
"""Daily Rotation V1 - aligned with Atropos backtest logic.
Backtest: +149% return, -3.3% DD, Sharpe 2.28."""
import csv, json, logging, os
from collections import defaultdict
from datetime import datetime, date, timedelta
import numpy as np
from xtquant import xtdata, xtconstant
from base_strategy_v2 import BaseStrategy
from execution.order_protocol import AmbiguousSubmissionError, OrderIntent, OrderStatus

logger = logging.getLogger("DailyRotationV1")
THEME_CSV = "D:/tool/dataset/kpl_all_sector_stocks_details.csv"
TEST_MODE = False


class DailyRotationV1(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.top_n = 15; self.max_position_pct = 0.07; self.cash_buffer = 0.08
        self.stop_loss_pct = 0.03; self.daily_loss_stop = 0.02; self.loss_freeze_days = 3
        self.portfolio_dd_stop = 0.12; self.portfolio_dd_freeze = 10
        self.is_test = "0"; self._last_trade_date = None; self._peak_nav = 0.0
        self._dd_frozen_until = None; self._daily_loss_frozen = None
        self._prev_nav = None; self._entry_prices = {}
        self._stock_themes = defaultdict(list)
        self._daily_nav_date = None  # date of last _prev_nav update (for overnight gap detection)
        self._load_theme_data()
        self._load_state()  # restore persisted state after defaults are set

    def _load_theme_data(self):
        if not os.path.exists(THEME_CSV): return
        with open(THEME_CSV, "r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                c = r.get("股票代码", "").strip(); t = r.get("题材名称", "").strip()
                if c and t: self._stock_themes[c].append(t)
        for k in self._stock_themes: self._stock_themes[k] = list(set(self._stock_themes[k]))
        logger.info(f"Themes: {len(self._stock_themes)} stocks")

    def _sf(self): return os.path.join(os.path.dirname(__file__), "daily_rotation_state.json")

    def _load_state(self):
        try:
            p = self._sf()
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f: s = json.load(f)
                self._peak_nav = s.get("peak_nav", 0.0); self._prev_nav = s.get("prev_nav")
                lt = s.get("last_trade_date")
                self._last_trade_date = date.fromisoformat(lt) if lt else None
                df = s.get("dd_frozen_until")
                self._dd_frozen_until = datetime.fromisoformat(df) if df else None
                dl = s.get("daily_loss_frozen")
                self._daily_loss_frozen = datetime.fromisoformat(dl) if dl else None
                self._entry_prices = s.get("entry_prices", {})
                nd = s.get("daily_nav_date")
                self._daily_nav_date = date.fromisoformat(nd) if nd else None
        except Exception as e: logger.warning(f"State load: {e}")

    def _save_state(self):
        try:
            s = {"peak_nav": self._peak_nav, "prev_nav": self._prev_nav,
                 "last_trade_date": self._last_trade_date.strftime("%Y-%m-%d") if self._last_trade_date else None,
                 "dd_frozen_until": self._dd_frozen_until.isoformat() if self._dd_frozen_until else None,
                 "daily_loss_frozen": self._daily_loss_frozen.isoformat() if self._daily_loss_frozen else None,
                 "daily_nav_date": self._daily_nav_date.strftime("%Y-%m-%d") if self._daily_nav_date else None,
                 "entry_prices": self._entry_prices}
            with open(self._sf(), "w", encoding="utf-8") as f: json.dump(s, f, ensure_ascii=False, indent=2)
        except Exception as e: logger.warning(f"State save: {e}")

    async def do(self, accounts):
        req = accounts.get("acc_1", accounts.get("acc_2", {}))
        xt_trader = req.get("xt_trader"); acc = req.get("account")
        self.is_test = req.get("is_test", "0")
        order_manager = req.get("order_manager") if self.is_test != "1" else None
        account_name = req.get("acc_name") or str(acc)
        if self.is_test != "1" and (
            order_manager is None
            or not req.get("managed_order_gate_open", False)
            or not getattr(order_manager, "submission_enabled", False)
        ):
            logger.error("[DailyRotationV1] managed order gate is closed")
            return json.dumps({"warn": [{"mark": "order_reconciliation_required"}]})
        if xt_trader is None or acc is None: return json.dumps({"warn": [{"mark": "no_account"}]})

        cur = datetime.now().time()
        if not TEST_MODE and self.is_test == "0" and cur < datetime.strptime("14:50", "%H:%M").time():
            return json.dumps({"msg": "not_time"})
        today = date.today()
        if not TEST_MODE and self._last_trade_date == today: return json.dumps({"msg": "done"})

        self._load_state()
        pv = self._pv(xt_trader, acc)
        if pv > self._peak_nav: self._peak_nav = pv; self._save_state()
        if self._peak_nav > 0:
            dd = (self._peak_nav - pv) / self._peak_nav
            if dd >= self.portfolio_dd_stop and self._dd_frozen_until is None:
                self._dd_frozen_until = datetime.now() + timedelta(days=self.portfolio_dd_freeze)
                self._save_state(); self._cash(xt_trader, acc, order_manager, account_name)
                logger.warning(f"DD STOP {dd:.2%}"); return json.dumps({"msg": "dd"})
        if self._dd_frozen_until:
            if datetime.now() < self._dd_frozen_until: return json.dumps({"msg": "dd"})
            self._dd_frozen_until = None; self._peak_nav = pv
        # Daily loss check — reset baseline on new trading day to avoid
        # overnight gap triggering the stop (e.g. stock gaps down 3% at open)
        new_day = self._daily_nav_date != today
        if new_day:
            self._daily_nav_date = today
            self._prev_nav = pv  # today's opening PV is the new baseline
            self._save_state()   # persist immediately so re-load in next do() call doesn't revert
        elif self._prev_nav and self._prev_nav > 0:
            dr = (pv - self._prev_nav) / self._prev_nav
            if dr < -self.daily_loss_stop and self._daily_loss_frozen is None:
                self._daily_loss_frozen = datetime.now() + timedelta(days=self.loss_freeze_days)
                self._save_state(); self._cash(xt_trader, acc, order_manager, account_name)
                logger.warning(f"DAY LOSS {dr:.2%}"); return json.dumps({"msg": "dayloss"})
        self._prev_nav = pv
        if self._daily_loss_frozen:
            if datetime.now() < self._daily_loss_frozen: return json.dumps({"msg": "dayloss"})
            self._daily_loss_frozen = None
        self._sl(xt_trader, acc, order_manager, account_name)

        ds = today.strftime("%Y%m%d")
        md, techs = self._load_data(ds)
        if not md: return json.dumps({"warn": [{"mark": "no_data"}]})

        # Market filters (aligned with backtest: MA20 breadth)
        if not self._mkt(md, techs):
            self._cash(xt_trader, acc, order_manager, account_name)
            return json.dumps({"msg": "bear"})

        sig = self._signal(md, techs)
        if len(sig) < self.top_n: return json.dumps({"warn": [{"mark": "few", "n": len(sig)}]})
        ranked = sorted(sig.items(), key=lambda x: x[1], reverse=True)
        sel = [c for c, _ in ranked[:self.top_n]]

        if TEST_MODE:
            logger.info(f"[TEST] PV={pv:,.0f} top5={sel[:5]} n={len(sel)} signals={len(sig)}")
            return json.dumps({"msg": "test", "top5": sel[:5], "n": len(sel)})

        self._cash(xt_trader, acc, order_manager, account_name)
        cash = self._cash_val(xt_trader, acc)
        pv2 = self._pv(xt_trader, acc)
        pos_val = min(cash * (1 - self.cash_buffer) / max(len(sel), 1), pv2 * self.max_position_pct)
        bought = 0
        for c in sel:
            price = self._px(c)
            if price <= 0:
                logger.info(f"SKIP {c}: no price")
                continue
            qty = int(pos_val / price / 100) * 100
            record = None
            if qty >= 200 and self.is_test != "1":
                record = self._submit_order(
                    xt_trader, acc, order_manager, account_name, c, "BUY", qty, price,
                    f"buy:{today.isoformat()}:{c}",
                )
                logger.info(f"BUY {c} qty={qty} price={price:.2f} order={record}")
                bought += 1
            else:
                logger.info(f"SKIP {c}: qty={qty} price={price:.2f} pos_val={pos_val:.0f} is_test={self.is_test}")
            if order_manager is None or (
                record is not None and record.filled_quantity > 0
            ):
                self._entry_prices[c] = float(
                    record.avg_fill_price if record is not None and record.avg_fill_price else price
                )
        self._last_trade_date = today; self._save_state()
        logger.info(f"DailyRot: {len(sel)} selected, {bought} bought, PV={pv:,.0f}, pos_val={pos_val:,.0f}")
        return json.dumps({"msg": "ok", "n": len(sel)})

    # ===== Data Loading (aligned with backtest) =====
    def _load_data(self, ds):
        all_s = []
        for s in ["沪深A股"]:
            try: all_s.extend(xtdata.get_stock_list_in_sector(s))
            except: pass
        all_s = list(set(all_s))
        logger.info(f"Loading {len(all_s)} stocks...")

        # Today's ticks
        ticks = {}
        for i in range(0, len(all_s), 500):
            try: ticks.update(xtdata.get_full_tick(all_s[i:i+500]))
            except: pass

        # Prev day close + 20-day daily bars for technicals
        prev_ds = (datetime.strptime(ds, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
        start_ds = (datetime.strptime(ds, "%Y%m%d") - timedelta(days=40)).strftime("%Y%m%d")
        daily_bars = {}
        for i in range(0, len(all_s), 300):
            batch = all_s[i:i+300]
            try:
                d = xtdata.get_market_data_ex(["close", "high", "low", "open", "volume"], batch, "1d", start_ds, ds)
            except: continue
            if not d: continue
            for code in batch:
                if code not in d or d[code] is None or len(d[code]) < 20: continue
                daily_bars[code] = d[code]

        # Build result
        r = {}
        techs = {}
        for code in all_s:
            tick = ticks.get(code, {})
            if not tick: continue
            try:
                c = float(tick.get("lastPrice", 0)); o = float(tick.get("open", 0))
                h = float(tick.get("high", 0)); l = float(tick.get("low", 0))
                v = float(tick.get("volume", 0))
            except: continue
            if c <= 0 or o <= 0: continue
            db = daily_bars.get(code)
            prev_c = float(db["close"].values[-2]) if db is not None and len(db) >= 2 else o
            r[code] = {"c": c, "o": o, "h": h, "l": l, "v": v, "pc": prev_c}
            if db is not None:
                closes = db["close"].values; highs = db["high"].values
                lows = db["low"].values; volumes = db["volume"].values
                # MA scores
                ma5 = np.mean(closes[-5:]); ma10 = np.mean(closes[-10:]); ma20 = np.mean(closes[-20:])
                ma_sc = (1 if closes[-1] > ma5 else 0) + (1 if closes[-1] > ma10 else 0) + (1 if closes[-1] > ma20 else 0) - 1.5
                # ATR (20d)
                tr = np.maximum(highs[-20:] - lows[-20:], np.abs(highs[-20:] - np.roll(closes, 1)[-20:]))
                atr = np.mean(tr) / closes[-1] if closes[-1] > 0 else 0.03
                # Vol ratio
                vol_5d = np.mean(volumes[-6:-1]) if len(volumes) >= 6 else volumes[-1]
                vr = volumes[-1] / vol_5d if vol_5d > 0 else 1.0
                # Market metrics for breadth
                above_ma = 1 if closes[-1] > ma20 else 0
                techs[code] = {"ma_sc": ma_sc, "atr": atr, "vr": vr, "above_ma": above_ma, "ret": closes[-1] / closes[-2] - 1 if len(closes) >= 2 else 0}
        logger.info(f"Loaded {len(r)} stocks, {len(techs)} with technicals")
        return r, techs

    # ===== Market Filters (aligned with backtest: >40% above MA20) =====
    def _mkt(self, md, techs):
        # 1. Breadth: % stocks above 20-day MA
        above = sum(1 for t in techs.values() if t.get("above_ma", 0) > 0)
        total = len(techs) if techs else 1
        breadth = above / total if total > 0 else 0
        if breadth < 0.40: return False
        # 2. Need at least 100 stocks
        return len(md) >= 100

    # ===== Signals (aligned with backtest) =====
    def _signal(self, md, techs):
        ts = self._theme_str(md)
        sig = {}
        for code, d in md.items():
            cl = d["c"]; op = d["o"]; dh = d["h"]; dl = d["l"]
            if op <= 0 or cl <= 0: continue

            # Quality filters (same as backtest)
            tech = techs.get(code, {})
            atr = tech.get("atr", 0.03); vr = tech.get("vr", 1.0)
            day_range = (dh - dl) / op
            if atr > 0.05 or atr < 0.005: continue
            if day_range > 0.10: continue
            if vr < 0.3: continue
            gap = op / d["pc"] - 1 if d["pc"] > 0 else 0
            if gap < -0.05: continue
            if cl / op - 1 < -0.09: continue

            # Theme strength (25% weight, same as backtest)
            themes = self._stock_themes.get(code.split(".")[0] if "." in code else code, [])
            tv = sum(ts.get(t, 0) for t in themes) / max(len(themes), 1)

            # Technical support (20%, same as backtest)
            ma_sc = tech.get("ma_sc", 0)

            # Gap-down recovery (15%)
            rec = cl / op - 1 if -0.05 < gap < -0.005 else 0

            # Price position (15%)
            pp = (cl - dl) / (dh - dl + 1e-8)
            ps = 1.0 - abs(pp - 0.65) * 2

            # VWAP distance (10%, proxy: price vs mid)
            mid = (dh + dl) / 2
            vp = cl / mid - 1 if mid > 0 else 0

            # Low reversal (10%)
            ld = min(cl / dl - 1, 0.05) if dl > 0 else 0

            score = (0.25 * tv * 100 + 0.20 * ma_sc + 0.15 * rec * 100 +
                     0.15 * ps + 0.10 * np.clip(vp * 100, -2, 2) + 0.10 * ld * 100 + 0.05 * vr)
            if score > -10:
                sig[code] = score  # All boards included (aligned with backtest)
        return sig

    def _theme_str(self, md):
        tr = defaultdict(list)
        for code, d in md.items():
            if d["o"] <= 0 or d["c"] <= 0: continue
            ret = d["c"] / d["o"] - 1
            clean = code.split(".")[0] if "." in code else code
            for t in self._stock_themes.get(clean, []): tr[t].append(ret)
        return {t: np.mean(rets) for t, rets in tr.items() if len(rets) >= 3}

    # ===== Trading helpers =====
    def _pv(self, xt_trader, acc):
        try:
            pv = sum(p.market_value for p in xt_trader.query_stock_positions(acc) if hasattr(p, "market_value") and p.market_value)
            a = xt_trader.query_stock_asset(acc); return (a.cash if a else 0) + pv
        except: return 1_000_000

    def _cash_val(self, xt_trader, acc):
        try: a = xt_trader.query_stock_asset(acc); return a.cash if a else 0
        except: return 0

    def _px(self, code):
        try:
            t = xtdata.get_full_tick([code])
            if t and code in t and t[code].get("lastPrice", 0):
                return float(t[code]["lastPrice"])
            # Try adding suffix if not found
            for sfx in [".SH", ".SZ", ".BJ"]:
                if "." in code: continue  # already has suffix
                c2 = code + sfx
                t2 = xtdata.get_full_tick([c2])
                if t2 and c2 in t2 and t2[c2].get("lastPrice", 0):
                    return float(t2[c2]["lastPrice"])
            return 0
        except: return 0

    def _pos(self, xt_trader, acc):
        r = {}
        try:
            for p in xt_trader.query_stock_positions(acc):
                if p.volume > 0:
                    r[p.stock_code] = {
                        "v": p.volume,
                        "open_price": float(getattr(p, "open_price", 0) or 0),
                    }
        except: pass
        return r

    def _cash(self, xt_trader, acc, order_manager=None, account_name=None):
        for c, p in self._pos(xt_trader, acc).items():
            if p["v"] <= 0: continue
            px = self._px(c)
            if px > 0 and self.is_test != "1":
                record = self._submit_order(xt_trader, acc, order_manager, account_name, c, "SELL", p["v"], px, f"cash:{date.today().isoformat()}:{c}")
                if order_manager is None or (record is not None and record.status is OrderStatus.FILLED):
                    self._entry_prices.pop(c, None)
        if order_manager is None:
            self._entry_prices.clear()

    def _sl(self, xt_trader, acc, order_manager=None, account_name=None):
        for c, p in self._pos(xt_trader, acc).items():
            if p["v"] <= 0: continue
            px = self._px(c); e = p.get("open_price") or self._entry_prices.get(c, 0)
            if e > 0 and (px / e - 1) < -self.stop_loss_pct:
                if self.is_test != "1":
                    record = self._submit_order(xt_trader, acc, order_manager, account_name, c, "SELL", p["v"], px, f"stop_loss:{date.today().isoformat()}:{c}")
                else:
                    record = None
                if order_manager is None or (record is not None and record.status is OrderStatus.FILLED):
                    self._entry_prices.pop(c, None)
                logger.info(f"SL {c}")

    @staticmethod
    def _submit_order(xt_trader, acc, order_manager, account_name, symbol, side,
                      quantity, price, signal_id):
        if order_manager is None:
            order_type = xtconstant.STOCK_BUY if side == "BUY" else xtconstant.STOCK_SELL
            return xt_trader.order_stock(
                acc, symbol, order_type, int(quantity), xtconstant.FIX_PRICE,
                float(price), "daily_rot",
            )
        intent = OrderIntent(
            account_id=str(account_name or acc), strategy_name="DailyRotationV1",
            symbol=symbol, side=side, quantity=int(quantity), order_type="LIMIT",
            limit_price=float(price), signal_id=signal_id,
            prediction_date=date.today().isoformat(), metadata={"strategy": "daily_rot"},
        )
        record = order_manager.submit(intent)
        if record.status in {OrderStatus.UNKNOWN, OrderStatus.REJECTED}:
            raise AmbiguousSubmissionError(
                f"DailyRotationV1 order {record.client_order_id} is {record.status.value}"
            )
        return record
