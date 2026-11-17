#coding=gbk
"""Dragon V6 - V20 backtest-optimized leader strategy for live trading.

Backtest: +63% return, -13% DD, Sharpe 4.0, Calmar 16.6 (2026-01~05)
Depends: dragon_v6_scores_*.json (daily 22:30 by export_v20_scores.py)
"""
import os
import sys
import json
import glob
import datetime
import logging
import numpy as np
from xtquant import xtdata
from xtquant import xtconstant
from base_strategy import BaseStrategy
from execution.order_protocol import AmbiguousSubmissionError, OrderIntent, OrderStatus

logger = logging.getLogger(__name__)

# V20 optimal parameters from backtest
CFG = {
    "top_n": 10,
    "max_position_pct": 0.03,
    "stop_loss_pct": 0.03,
    "dd_stop_l1": 0.10,
    "dd_stop_l2": 0.20,
    "dd_freeze_days": 5,
    "crash_sl_threshold": 4,
    "daily_loss_circuit": 0.07,
    "rebalance_days": 5,
    "lot_size": 100,
}


class DragonV6(BaseStrategy):
    """V20 dragon leader strategy for QMT live trading."""

    def __init__(self, config):
        super().__init__(config)
        self.cfg = CFG.copy()

        # State file for persistence across restarts
        base_dir = config.get('strategy', 'base_dir', fallback='D:/tool/dataset/strategy_data')
        self._state_file = os.path.join(base_dir, 'DragonV6_state.json')

        # Load persisted state (or defaults)
        self._position_highs = {}
        self._peak_nav = 0.0
        self._trailing_peak = 0.0
        self._dd_level = 0
        self._days_since_rebalance = 999
        self._last_rebalance_date = None
        self._prev_nav = 0.0
        self._sl_count_today = 0
        self._sl_history = []
        self._daily_rets = []
        self._dd_frozen_until = None
        self._recovery_step = 0
        self._load_state()

        # Model scores
        self._monthly_scores = {}
        self._snap_dates = []
        self._score_floor = 0.0
        self._scores_file = ""
        self._load_scores()

    def _load_state(self):
        """Restore persisted state from disk (survives daily restart)."""
        if os.path.exists(self._state_file):
            try:
                with open(self._state_file, 'r') as f:
                    s = json.load(f)
                self._peak_nav = s.get('peak_nav', 0.0)
                self._trailing_peak = s.get('trailing_peak', 0.0)
                self._dd_level = s.get('dd_level', 0)
                self._position_highs = s.get('position_highs', {})
                lrb = s.get('last_rebalance_date')
                self._last_rebalance_date = datetime.date.fromisoformat(lrb) if lrb else None
                print("[DragonV6] State restored: peak=%.0f, dd_level=%d, highs=%d stocks" % (
                    self._peak_nav, self._dd_level, len(self._position_highs)))
            except Exception as e:
                print("[DragonV6] Failed to load state: %s" % e)

    def _save_state(self):
        """Persist critical state to disk."""
        try:
            s = {
                'peak_nav': self._peak_nav,
                'trailing_peak': self._trailing_peak,
                'dd_level': self._dd_level,
                'position_highs': self._position_highs,
                'last_rebalance_date': str(self._last_rebalance_date) if hasattr(self, '_last_rebalance_date') and self._last_rebalance_date else None,
                'updated': datetime.datetime.now().isoformat(),
            }
            with open(self._state_file, 'w') as f:
                json.dump(s, f)
        except Exception as e:
            print("[DragonV6] Failed to save state: %s" % e)

    def _load_scores(self):
        """Load latest monthly snapshot scores from JSON."""
        scores_dir = self.config.get('DragonV6', 'scores_dir', fallback='./logs')
        pattern = os.path.join(scores_dir, 'dragon_v6_scores_*.json')
        files = sorted(glob.glob(pattern))
        if not files:
            print("[DragonV6] WARN: No score files found in %s" % scores_dir)
            return
        self._scores_file = files[-1]
        with open(self._scores_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self._monthly_scores = data.get('scores', {})
        self._snap_dates = sorted(self._monthly_scores.keys())
        all_scores = []
        for ss in self._monthly_scores.values():
            all_scores.extend(ss.values())
        if all_scores:
            self._score_floor = float(np.percentile(all_scores, 50))
        print("[DragonV6] Loaded %d snapshots, %d stocks, floor=%.3f" % (
            len(self._snap_dates), len(all_scores), self._score_floor))

    def _get_current_scores(self):
        """Get scores from most recent snapshot <= today."""
        today = datetime.date.today()
        best = None
        for sd in self._snap_dates:
            d = datetime.date.fromisoformat(sd)
            if d <= today:
                best = sd
        if best:
            return self._monthly_scores.get(best, {})
        return {}

    async def do(self, accounts):
        """Main entry - called periodically by Hermes engine."""
        req = accounts.get("acc_1", accounts.get("acc_2", {}))
        xt_trader = req.get("xt_trader")
        acc = req.get("account")
        is_test = req.get("is_test", "0")
        order_manager = req.get("order_manager") if is_test != "1" else None
        account_name = req.get("acc_name") or str(acc)
        if is_test != "1" and (
            order_manager is None
            or not req.get("managed_order_gate_open", False)
            or not getattr(order_manager, "submission_enabled", False)
        ):
            logger.error("[DragonV6] managed order gate is closed; reconciliation is required")
            return json.dumps({"msg": [{"mark": "order_reconciliation_required"}]})

        # Check trading hours
        cur = datetime.datetime.now().time()
        s = datetime.datetime.strptime("09:30", "%H:%M").time()
        e = datetime.datetime.strptime("14:55", "%H:%M").time()
        m1 = datetime.datetime.strptime("11:30", "%H:%M").time()
        m2 = datetime.datetime.strptime("13:01", "%H:%M").time()
        ok = (s <= cur <= m1) or (m2 <= cur <= e)
        if not ok and is_test == "0":
            return json.dumps({"msg": [{"mark": "not trade time"}]})

        logger.info("[DragonV6] do() snap=%d floor=%.3f", len(self._snap_dates), self._score_floor)

        if xt_trader is None or acc is None:
            logger.warning("[DragonV6] xt_trader/acc is None!")
            return json.dumps({"msg": [{"mark": "no_connection"}]})

        # Current positions (strip suffix for internal use)
        pos_map = {}
        for p in xt_trader.query_stock_positions(acc):
            if p.volume > 0:
                raw = p.stock_code.split('.')[0]
                pos_map[raw] = {
                    'vol': p.volume, 'cost': p.open_price, 'mv': p.market_value,
                    'qmt_code': p.stock_code
                }
        asset = xt_trader.query_stock_asset(acc)
        cash = asset.cash if asset else 0
        tv = cash + sum(x['mv'] for x in pos_map.values())
        # Check for pending orders (avoid duplicate buys on restart)
        pending_orders = xt_trader.query_stock_orders(acc, True)
        has_pending = any(o.stock_code for o in pending_orders) if pending_orders else False
        # Auto-heal: if NAV recovered above peak, reset DD state
        if self._peak_nav > 0 and tv > self._peak_nav * 0.95 and self._dd_level > 0:
            logger.info("[DragonV6] NAV recovered to %.0f (peak was %.0f), resetting DD", tv, self._peak_nav)
            self._dd_level = 0; self._peak_nav = tv; self._trailing_peak = tv

        logger.info("[DragonV6] Account cash=%.0f, tv=%.0f, pos=%d, pending=%d", cash, tv, len(pos_map),
                    1 if has_pending else 0)
        # Force initial entry only once, when never traded before
        if not pos_map and not has_pending and self._last_rebalance_date is None:
            self._days_since_rebalance = 999
        if self._prev_nav == 0:
            self._prev_nav = tv
        if tv > self._peak_nav:
            self._peak_nav = tv
        if tv > self._trailing_peak:
            self._trailing_peak = tv
            # Reset DD level on new high
            if self._dd_level > 0:
                self._dd_level = 0
                print("[DragonV6] New high, DD level reset")

        # Get scores and candidates
        scores = self._get_current_scores()
        if not scores:
            return json.dumps({"msg": [{"mark": "no scores"}]})
        ranked = sorted(
            [(c, s) for c, s in scores.items() if s >= self._score_floor],
            key=lambda x: x[1], reverse=True)
        selected = [c for c, _ in ranked[:self.cfg['top_n']]]

        orders = self._apply_rules(
            xt_trader, acc, pos_map, selected, scores, cash, tv,
            order_manager=order_manager, account_name=account_name,
        )
        self._prev_nav = tv
        self._save_state()
        return json.dumps({"msg": orders})

    @staticmethod
    def _add_suffix(code):
        """Add exchange suffix for QMT: 6xxxxx->.SH, 0xxxxx/3xxxxx->.SZ, 8xxxxx->.BJ"""
        if '.' in code:
            return code
        if code.startswith('6'):
            return code + '.SH'
        elif code.startswith('0') or code.startswith('3'):
            return code + '.SZ'
        elif code.startswith('8') or code.startswith('4'):
            return code + '.BJ'
        return code + '.SH'

    def _apply_rules(self, xt_trader, acc, pos_map, selected, scores, cash, tv,
                     *, order_manager=None, account_name=None):
        """Apply all V20 risk rules and generate orders."""
        self._sl_count_today = 0
        orders = []
        sold_today = set()
        logger.info("[DragonV6] _apply_rules: selected=%d, pos=%d, cash=%.0f, tv=%.0f, dd=%d, days=%d",
                    len(selected), len(pos_map), cash, tv, self._dd_level, self._days_since_rebalance)

        # Build price snapshot using real-time tick data (day data not available during hours)
        prices = {}
        all_raw = list(set(selected + list(pos_map.keys())))
        all_syms = [self._add_suffix(s) for s in all_raw]
        logger.info("[DragonV6] Querying prices for %d stocks: %s...", len(all_syms), str(all_syms[:3]))
        qmt_to_raw = {self._add_suffix(r): r for r in all_raw}
        try:
            # Use subscribe_whole_quote for real-time data
            tick_data = xtdata.get_full_tick(all_syms)
            if tick_data:
                for qmt_code in all_syms:
                    if qmt_code in tick_data:
                        tick = tick_data[qmt_code]
                        px = tick.get('lastPrice', 0) or tick.get('lastClose', 0)
                        if px and float(px) > 0:
                            prices[qmt_to_raw.get(qmt_code, qmt_code)] = float(px)
            logger.info("[DragonV6] Full tick price OK: %d/%d", len(prices), len(all_syms))
        except Exception as e:
            logger.warning("[DragonV6] Full tick failed: %s, trying market_data...", e)
            # Fallback: try get_market_data_ex
            try:
                tick_data = xtdata.get_market_data_ex([], all_syms, period='1d')
                if tick_data:
                    for qmt_code in all_syms:
                        if qmt_code in tick_data:
                            px = tick_data[qmt_code].get('lastPrice', 0)
                            if px and float(px) > 0:
                                prices[qmt_to_raw.get(qmt_code, qmt_code)] = float(px)
            except Exception:
                pass
            logger.info("[DragonV6] Fallback price OK: %d/%d", len(prices), len(all_syms))

        # 1. Daily loss circuit breaker (7%) -- skip if no settled positions (cash deployment)
        if self._prev_nav > 0 and len(pos_map) > 0:
            dr = (tv - self._prev_nav) / self._prev_nav
            if dr < -self.cfg['daily_loss_circuit']:
                logger.warning("[DragonV6] Daily loss %.1f%%, circuit breaker!", dr * 100)
                orders.extend(self._reduce(xt_trader, acc, pos_map, prices, 0.50,
                                           order_manager=order_manager, account_name=account_name))
                return orders

        # 2. DD graduated response
        if self._trailing_peak > 0:
            dd = (self._trailing_peak - tv) / self._trailing_peak
            if dd >= self.cfg['dd_stop_l2'] and self._dd_level < 2:
                self._dd_level = 2
                print("[DragonV6] DD L2 %.1f%%, cash" % (dd * 100))
                orders.extend(self._reduce(xt_trader, acc, pos_map, prices, 0.0,
                                           order_manager=order_manager, account_name=account_name))
                return orders
            elif dd >= self.cfg['dd_stop_l1'] and self._dd_level == 0:
                self._dd_level = 1
                print("[DragonV6] DD L1 %.1f%%, halve" % (dd * 100))
                orders.extend(self._reduce(xt_trader, acc, pos_map, prices, 0.50,
                                           order_manager=order_manager, account_name=account_name))

        # 3. Trailing stop loss (3% from high)
        for sym, pos in pos_map.items():
            px = prices.get(sym, 0)
            if px <= 0 or pos['vol'] <= 0:
                continue
            prev_high = self._position_highs.get(sym, 0)
            if px > prev_high:
                self._position_highs[sym] = px
                continue
            high = self._position_highs.get(sym, pos['cost'])
            if high > 0 and (px / high - 1) < -self.cfg['stop_loss_pct']:
                qmt_code = pos.get('qmt_code', self._add_suffix(sym))
                sell_price = round(px * 0.99, 2)
                self._submit_order(xt_trader, acc, order_manager, account_name,
                                   qmt_code, "SELL", pos['vol'], sell_price, "stop_loss")
                orders.append({
                    'code': qmt_code, 'price': sell_price,
                    'action': 'sell', 'volume': pos['vol'], 'reason': 'sl'
                })
                self._sl_count_today += 1
                self._position_highs.pop(sym, None)
                sold_today.add(sym)

        # 4. Crash detection (>=4 SLs in one cycle)
        if self._sl_count_today >= self.cfg['crash_sl_threshold']:
            print("[DragonV6] CRASH %d SLs, halving" % self._sl_count_today)
            orders.extend(self._reduce(xt_trader, acc, pos_map, prices, 0.50,
                                       order_manager=order_manager, account_name=account_name))

        # 5. Rebalance (date-based, not call-count-based)
        today_date = datetime.date.today()
        last_rb = getattr(self, '_last_rebalance_date', None)
        if last_rb is None or (today_date - last_rb).days >= self.cfg['rebalance_days']:
            sel_set = set(selected)
            # Sell non-selected (skip already-sold via SL/reduce)
            for sym in list(pos_map.keys()):
                if sym in sold_today:
                    continue
                if sym not in sel_set and pos_map[sym]['vol'] > 0:
                    px = prices.get(sym, 0)
                    if px > 0:
                        qmt_code = pos_map[sym].get('qmt_code', self._add_suffix(sym))
                        sell_price = round(px * 0.99, 2)
                        self._submit_order(xt_trader, acc, order_manager, account_name,
                                           qmt_code, "SELL", pos_map[sym]['vol'], sell_price,
                                           "rebalance_sell")
                        orders.append({
                            'code': qmt_code, 'price': sell_price,
                            'action': 'sell', 'volume': pos_map[sym]['vol'],
                            'reason': 'rebalance'
                        })
                        sold_today.add(sym)
            # Buy selected (score-weighted)
            ws = np.array([scores.get(s, 0) for s in selected], dtype=np.float64)
            wmin = ws.min()
            wr = ws.max() - wmin + 1e-10
            wts = (0.5 + (ws - wmin) / wr)
            wts = wts / wts.sum()
            exp = (1.0 if self._dd_level == 0 else 0.50) * 0.90
            for i, sym in enumerate(selected):
                if sym in pos_map:
                    continue
                px = prices.get(sym, 0)
                if px <= 0:
                    continue
                tgt = min(tv * exp * wts[i], tv * self.cfg['max_position_pct'])
                qty = int(tgt / px / 100) * 100
                if qty < 100 or cash < qty * px * 1.005:
                    continue
                qmt_code = self._add_suffix(sym)
                self._submit_order(xt_trader, acc, order_manager, account_name,
                                   qmt_code, "BUY", qty, px, "rebalance_buy")
                orders.append({
                    'code': qmt_code, 'price': px, 'action': 'buy',
                    'volume': qty, 'reason': 'rebalance'
                })
                cash -= qty * px * 1.005
                self._position_highs[sym] = px
            self._last_rebalance_date = today_date
        logger.info("[DragonV6] _apply_rules done: %d orders generated", len(orders))
        return orders

    def _reduce(self, xt_trader, acc, pos_map, prices, keep, *, order_manager=None, account_name=None):
        """Reduce all positions to keep fraction of current volume."""
        o = []
        for sym, pos in pos_map.items():
            if pos['vol'] <= 0:
                continue
            sq = int(pos['vol'] * (1 - keep) / 100) * 100
            if sq > 0:
                px = prices.get(sym, pos['cost'])
                qmt_code = pos.get('qmt_code', self._add_suffix(sym))
                sell_price = round(px * 0.99, 2)
                if keep == 0:
                    self._submit_order(xt_trader, acc, order_manager, account_name,
                                       qmt_code, "SELL", pos['vol'], sell_price, "reduce_all")
                else:
                    self._submit_order(xt_trader, acc, order_manager, account_name,
                                       qmt_code, "SELL", sq, sell_price, "reduce")
                o.append({
                    'code': qmt_code, 'price': sell_price,
                    'action': 'sell', 'volume': sq, 'reason': 'reduce'
                })
        return o

    @staticmethod
    def _submit_order(xt_trader, acc, order_manager, account_name, symbol, side,
                      quantity, price, reason):
        if int(quantity) <= 0:
            return None
        if order_manager is None:
            order_type = xtconstant.STOCK_BUY if side == "BUY" else xtconstant.STOCK_SELL
            return xt_trader.order_stock(
                acc, symbol, order_type, int(quantity), xtconstant.LATEST_PRICE,
                float(price), strategy_name="dragon_v6",
            )
        today = datetime.date.today().isoformat()
        intent = OrderIntent(
            account_id=str(account_name or acc), strategy_name="DragonV6", symbol=symbol,
            side=side, quantity=int(quantity), order_type="LIMIT", limit_price=float(price),
            signal_id=f"{today}:{reason}:{symbol}:{side}:{int(quantity)}",
            signal_timestamp=datetime.datetime.now().isoformat(), metadata={"reason": reason},
        )
        record = order_manager.submit(intent)
        if record.status in {OrderStatus.UNKNOWN, OrderStatus.REJECTED}:
            raise AmbiguousSubmissionError(
                f"DragonV6 order {record.client_order_id} is {record.status.value}"
            )
        return record


if __name__ == "__main__":
    import configparser
    a = DragonV6(config=configparser.ConfigParser())
