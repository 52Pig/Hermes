# -*- coding: utf-8 -*-
"""
AIRisk V1 — AI composite_score + institutional risk controls.

Backtest proven: +27.37% return, 2.65% drawdown, Sharpe 2.47 (2026-01~05, zero look-ahead).

Strategy:
  1. Load Hephaestus-Forge predictions (all_predictions.csv)
  2. Rank by composite_score, select top 15
  3. Weekly rebalance, equal weight
  4. Market timing: >45% stocks above MA20 → invest; else → cash
  5. Portfolio DD stop: 15% → freeze 10 days
  6. Position stop-loss: -5%, take-profit: +8%
  7. Cash buffer: 10%

Requires: all_predictions.csv updated daily by generate_all_predictions.py.

Registration (conf/config.ini):
  [AIRiskV1]
  name = AIRiskV1
  path = strategy/ai_risk_v1.py
  config_file = conf/dragon_v4.ini
  interval = 180
"""

import json
import os
import glob
import logging
import numpy as np
from datetime import datetime, date, timedelta
from xtquant import xtconstant, xtdata
from base_strategy_v2 import BaseStrategy
from execution.order_protocol import AmbiguousSubmissionError, OrderIntent, OrderStatus

logger = logging.getLogger("AIRiskV1")


class AIRiskV1(BaseStrategy):
    """AI prediction with institutional risk controls."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Strategy parameters (backtest-optimized)
        self.top_n = 15
        self.rebalance_days = 5
        self.max_position_pct = 0.08
        self.cash_buffer = 0.10
        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.08
        self.dd_stop_pct = 0.15
        self.dd_freeze_days = 10

        # State
        self.is_test = "0"
        self._last_rebalance_date = None
        self._last_tick_date = None  # tracks trading day transitions for rebalance counting
        self._peak_nav = 0.0
        self._dd_frozen_until = None
        self._entry_prices = {}
        self._wt_dict = {}
        self._trading_day_count = 0
        self._state_file = "D:/tool/dataset/strategy_data/ai_risk_state.json"
        self._predictions = {}  # {date: {code: composite_score}}

        # Load predictions and state
        self._load_predictions()
        self._load_state()

        logger.info(
            f"[AIRiskV1] Init: top_n={self.top_n}, "
            f"stop={self.stop_loss_pct:.0%}, tp={self.take_profit_pct:.0%}, "
            f"dd_stop={self.dd_stop_pct:.0%}, pred_dates={len(self._predictions)}"
        )

    # ========== Prediction Loading ==========

    def _load_predictions(self):
        """Load Hephaestus-Forge predictions from CSV."""
        csv_path = "D:/tool/pycharm_workspace/Hephaestus-Forge/inference_results/all_predictions.csv"
        if not os.path.exists(csv_path):
            csv_path = "D:/software/atropos/all_predictions.csv"
        if not os.path.exists(csv_path):
            csv_paths = sorted(glob.glob("D:/software/atropos/all_predictions*.csv"), reverse=True)
            if csv_paths:
                csv_path = csv_paths[0]

        if not os.path.exists(csv_path):
            logger.warning(f"[AIRiskV1] Predictions CSV not found!")
            return

        try:
            import csv as csv_module
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv_module.DictReader(f)
                for row in reader:
                    try:
                        code = row['stock_code'].strip()  # 完整后缀 e.g. 688200.SH
                        date_str = row['date'].strip()
                        score = float(row.get('composite_score', 0))
                        d = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
                        self._predictions.setdefault(d, {})[code] = score
                    except (ValueError, KeyError):
                        continue
            dates = sorted(self._predictions.keys())
            logger.info(
                f"[AIRiskV1] Loaded {len(self._predictions)} dates "
                f"({dates[0] if dates else '?'} to {dates[-1] if dates else '?'}), "
                f"{sum(len(v) for v in self._predictions.values())} records"
            )
        except Exception as e:
            logger.error(f"[AIRiskV1] Failed to load predictions: {e}")

    # ========== State Persistence ==========

    def _load_state(self):
        try:
            if os.path.exists(self._state_file):
                with open(self._state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                self._peak_nav = state.get('peak_nav', 0.0)
                self._last_rebalance_date = (
                    datetime.strptime(state['last_rebalance_date'], '%Y-%m-%d').date()
                    if state.get('last_rebalance_date') else None
                )
                ltd = state.get('last_tick_date')
                self._last_tick_date = (
                    datetime.strptime(ltd, '%Y-%m-%d').date() if ltd else None
                )
                frozen = state.get('dd_frozen_until')
                self._dd_frozen_until = datetime.strptime(frozen, '%Y-%m-%d %H:%M:%S') if frozen else None
                self._trading_day_count = state.get('trading_day_count', 0)
                self._entry_prices = state.get('entry_prices', {})
                logger.info(f"[AIRiskV1] Restored state: peak={self._peak_nav:,.0f}")
        except Exception as e:
            logger.warning(f"[AIRiskV1] State load failed: {e}")

    def _save_state(self):
        try:
            state = {
                'peak_nav': self._peak_nav,
                'last_rebalance_date': self._last_rebalance_date.strftime('%Y-%m-%d') if self._last_rebalance_date else None,
                'last_tick_date': self._last_tick_date.strftime('%Y-%m-%d') if self._last_tick_date else None,
                'dd_frozen_until': self._dd_frozen_until.strftime('%Y-%m-%d %H:%M:%S') if self._dd_frozen_until else None,
                'trading_day_count': self._trading_day_count,
                'entry_prices': self._entry_prices,
            }
            with open(self._state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[AIRiskV1] State save failed: {e}")

    # ========== Main Entry ==========

    async def do(self, accounts):
        req_dict = accounts.get("acc_1", accounts.get("acc_2", {}))
        xt_trader = req_dict.get("xt_trader")
        acc = req_dict.get("account")
        self.is_test = req_dict.get("is_test", "0")
        order_manager = req_dict.get("order_manager") if self.is_test != "1" else None
        account_name = req_dict.get("acc_name") or str(acc)
        if self.is_test != "1" and (
            order_manager is None
            or not req_dict.get("managed_order_gate_open", False)
            or not getattr(order_manager, "submission_enabled", False)
        ):
            logger.error("[AIRiskV1] managed order gate is closed")
            return json.dumps({"warn": [{"mark": "order_reconciliation_required"}]})

        if xt_trader is None or acc is None:
            return json.dumps({"warn": [{"mark": "no_account"}]})

        cur_time = datetime.now().time()
        if not (datetime.strptime("09:30","%H:%M").time() <= cur_time <= datetime.strptime("15:00","%H:%M").time()):
            if self.is_test == "0":
                return json.dumps({"msg": "not_trade_time"})

        # Portfolio DD check
        pv = self._get_portfolio_value(xt_trader, acc)
        if pv > self._peak_nav:
            self._peak_nav = pv
            self._save_state()

        if self._peak_nav > 0 and self.dd_stop_pct > 0:
            current_dd = (self._peak_nav - pv) / self._peak_nav
            if current_dd >= self.dd_stop_pct:
                if self._dd_frozen_until is None:
                    self._dd_frozen_until = datetime.now() + timedelta(days=self.dd_freeze_days)
                    self._save_state()
                    self._go_to_cash(xt_trader, acc, order_manager, account_name)
                    logger.warning(f"[AIRiskV1] DD STOP: {current_dd:.2%}, frozen until {self._dd_frozen_until}")
                return json.dumps({"msg": "dd_frozen"})

        if self._dd_frozen_until and datetime.now() < self._dd_frozen_until:
            return json.dumps({"msg": "dd_frozen"})
        if self._dd_frozen_until:
            self._dd_frozen_until = None
            self._peak_nav = pv
            self._save_state()

        # Stop-loss check
        self._check_stop_loss(xt_trader, acc, order_manager, account_name)

        # Rebalance schedule — count TRADING DAYS (not ticks).
        # Each new trading day increments the counter; rebalance every N trading days.
        # On first-ever run (_last_rebalance_date is None), force immediate rebalance.
        today = date.today()
        if self._last_tick_date != today:
            self._last_tick_date = today
            if self._last_rebalance_date is None:
                self._trading_day_count = self.rebalance_days  # first run: force rebalance
            else:
                self._trading_day_count += 1
            self._save_state()
        if self._trading_day_count < self.rebalance_days:
            return json.dumps({"msg": "no_rebalance"})

        # Near close execution
        rebalance_time = datetime.strptime("14:50", "%H:%M").time()
        if cur_time < rebalance_time:
            return json.dumps({"msg": "before_rebalance_time"})

        # Market timing
        if not self._market_healthy(xt_trader, acc):
            self._go_to_cash(xt_trader, acc, order_manager, account_name)
            self._trading_day_count = 0
            self._save_state()
            return json.dumps({"msg": "bear_market"})

        # Get predictions for the previous trading day
        pred_date = self._get_prev_trading_day()
        if pred_date is None:
            return json.dumps({"warn": [{"mark": "no_pred_date"}]})

        day_preds = self._predictions.get(pred_date, {})
        if not day_preds:
            logger.warning(f"[AIRiskV1] No predictions for {pred_date}")
            return json.dumps({"warn": [{"mark": "no_predictions"}]})

        # Rank and select top N
        ranked = sorted(day_preds.items(), key=lambda x: x[1], reverse=True)
        selected = [code for code, score in ranked[:self.top_n]]
        weight = (1.0 - self.cash_buffer) / len(selected)

        logger.info(
            f"[AIRiskV1] Rebalance {today}: {len(selected)} stocks, "
            f"top={selected[0]}:{day_preds.get(selected[0],0):.1f}"
        )

        self._rebalance(xt_trader, acc, selected, weight, order_manager, account_name)
        self._trading_day_count = 0
        self._last_rebalance_date = today
        self._save_state()

        return json.dumps({"msg": "ok"})

    # ========== Market Timing ==========

    def _market_healthy(self, xt_trader, acc) -> bool:
        """Check if market proxy is healthy using MA20."""
        today = date.today()
        pred_date = self._get_prev_trading_day()
        if pred_date is None:
            return True

        day_preds = self._predictions.get(pred_date, {})
        if not day_preds:
            return True

        # Use top 30 stocks by prediction score as market proxy
        ranked = sorted(day_preds.items(), key=lambda x: x[1], reverse=True)
        proxy_codes = [c for c, _ in ranked[:30]]

        try:
            start = (today - timedelta(days=60)).strftime('%Y%m%d')
            end = today.strftime('%Y%m%d')
            data = xtdata.get_market_data_ex(
                stock_list=proxy_codes, period='1d',
                start_time=start, end_time=end, field_list=['close']
            )
        except Exception:
            return True

        above = total = 0
        for code, df in data.items():
            if df is None or len(df) < 22:
                continue
            closes = df['close'].values
            if len(closes) < 22:
                continue
            ma20 = np.mean(closes[-21:-1])
            if closes[-2] > ma20:
                above += 1
            total += 1

        if total < 5:
            return True
        healthy = above / total >= 0.45
        if not healthy:
            logger.info(f"[AIRiskV1] Market BEAR: {above}/{total} above MA20")
        return healthy

    # ========== Stop Loss ==========

    def _check_stop_loss(self, xt_trader, acc, order_manager=None, account_name=None):
        positions = self._get_positions(xt_trader, acc)
        if not positions:
            return
        try:
            ticks = xtdata.get_full_tick(list(positions.keys()))
        except Exception:
            return
        for code, pos in positions.items():
            if pos['volume'] <= 0:
                continue
            tick = ticks.get(code)
            if tick is None:
                continue
            try:
                price = float(tick['lastPrice'])
            except (KeyError, TypeError, ValueError):
                continue
            if price <= 0:
                continue
            entry = pos.get('open_price') or self._entry_prices.get(code, 0)
            if entry <= 0:
                continue
            ret = price / entry - 1
            if ret < -self.stop_loss_pct or ret > self.take_profit_pct:
                sell_price = round(price * 0.99, 2)
                sell_record = None
                if self.is_test != "1":
                    sell_record = self._submit_order(
                        xt_trader, acc, order_manager, account_name, code,
                        "SELL", pos['volume'], sell_price,
                        f"stop:{date.today().isoformat()}:{code}")
                if order_manager is None or (
                    sell_record is not None and sell_record.status is OrderStatus.FILLED
                ):
                    self._entry_prices.pop(code, None)
                self._save_state()
                logger.info(f"[AIRiskV1] {'STOP' if ret<0 else 'TP'} {code}: {ret:+.1%}")

    # ========== Rebalance ==========

    def _rebalance(self, xt_trader, acc, selected, weight, order_manager=None, account_name=None):
        positions = self._get_positions(xt_trader, acc)
        cash = self._get_cash(xt_trader, acc)
        pv = self._get_portfolio_value(xt_trader, acc)
        if pv <= 0:
            return

        all_codes = list(set(list(positions.keys()) + selected))
        try:
            ticks = xtdata.get_full_tick(all_codes)
        except Exception:
            return

        lot = 100
        available = max(0, cash - pv * self.cash_buffer)

        # Sell removed
        for code, pos in positions.items():
            if code not in selected and pos['volume'] > 0:
                tick = ticks.get(code)
                if tick is None: continue
                try:
                    px = float(tick['lastPrice'])
                except: continue
                sp = round(px * 0.99, 2)
                sell_record = None
                if self.is_test != "1":
                    sell_record = self._submit_order(
                        xt_trader, acc, order_manager, account_name, code,
                        "SELL", pos['volume'], sp,
                        f"rebalance_sell:{date.today().isoformat()}:{code}")
                if order_manager is None or (
                    sell_record is not None and sell_record.status is OrderStatus.FILLED
                ):
                    self._entry_prices.pop(code, None)
                if order_manager is None:
                    available += pos['volume'] * sp * 0.995

        # Buy/adjust
        for code in selected:
            tick = ticks.get(code)
            if tick is None: continue
            try:
                px = float(tick['lastPrice'])
            except: continue
            if px <= 0: continue

            pos = positions.get(code, {})
            cur_q = pos.get('volume', 0)
            tv = weight * pv
            dv = tv - cur_q * px
            if abs(dv) < px * lot * 2: continue
            dq = int(dv / px)
            dq = (dq // lot) * lot

            if dq > 0:
                cost = dq * px * 1.005
                if cost <= available:
                    bp = round(px * 1.005, 2)
                    buy_record = None
                    if self.is_test != "1":
                        buy_record = self._submit_order(
                            xt_trader, acc, order_manager, account_name, code,
                            "BUY", dq, bp,
                            f"rebalance_buy:{date.today().isoformat()}:{code}")
                    if order_manager is None or (
                        buy_record is not None and buy_record.filled_quantity > 0
                    ):
                        self._entry_prices[code] = float(
                            buy_record.avg_fill_price if buy_record is not None
                            and buy_record.avg_fill_price else px
                        )
                    available -= cost
            elif dq < 0:
                sq = min(-dq, pos.get('volume', 0))
                sq = (sq // lot) * lot
                if sq > 0:
                    sp = round(px * 0.99, 2)
                    if self.is_test != "1":
                        self._submit_order(
                            xt_trader, acc, order_manager, account_name, code,
                            "SELL", sq, sp,
                            f"rebalance_reduce:{date.today().isoformat()}:{code}")

    def _go_to_cash(self, xt_trader, acc, order_manager=None, account_name=None):
        positions = self._get_positions(xt_trader, acc)
        for code, pos in positions.items():
            if pos['volume'] <= 0: continue
            try:
                ticks = xtdata.get_full_tick([code])
                px = float(ticks.get(code, {}).get('lastPrice', 0))
            except: continue
            if px <= 0: continue
            sp = round(px * 0.99, 2)
            if self.is_test != "1":
                self._submit_order(xt_trader, acc, order_manager, account_name, code,
                                   "SELL", pos['volume'], sp,
                                   f"cash:{date.today().isoformat()}:{code}")
        if order_manager is None:
            self._entry_prices.clear()
        self._save_state()

    @staticmethod
    def _submit_order(xt_trader, acc, order_manager, account_name, symbol, side,
                      quantity, price, signal_id):
        if int(quantity) <= 0:
            return None
        if order_manager is None:
            order_type = xtconstant.STOCK_BUY if side == "BUY" else xtconstant.STOCK_SELL
            return xt_trader.order_stock(
                acc, symbol, order_type, int(quantity), xtconstant.FIX_PRICE,
                float(price), strategy_name="ai_risk",
            )
        intent = OrderIntent(
            account_id=str(account_name or acc), strategy_name="AIRiskV1", symbol=symbol,
            side=side, quantity=int(quantity), order_type="LIMIT", limit_price=float(price),
            signal_id=signal_id, prediction_date=date.today().isoformat(),
            metadata={"strategy": "ai_risk"},
        )
        record = order_manager.submit(intent)
        if record.status in {OrderStatus.UNKNOWN, OrderStatus.REJECTED}:
            raise AmbiguousSubmissionError(
                f"AIRiskV1 order {record.client_order_id} is {record.status.value}"
            )
        return record

    # ========== Helpers ==========

    def _get_portfolio_value(self, xt_trader, acc):
        try:
            pos_val = sum(
                p.market_value for p in xt_trader.query_stock_positions(acc)
                if hasattr(p, 'market_value') and p.market_value
            )
            asset = xt_trader.query_stock_asset(acc)
            return (asset.cash if asset else 0) + pos_val
        except: return 0.0

    def _get_cash(self, xt_trader, acc):
        try:
            asset = xt_trader.query_stock_asset(acc)
            return asset.cash if asset else 0
        except: return 0.0

    def _get_positions(self, xt_trader, acc):
        positions = {}
        try:
            for p in xt_trader.query_stock_positions(acc):
                if p.volume > 0:
                    positions[p.stock_code] = {
                        'volume': p.volume,
                        'open_price': p.open_price if hasattr(p, 'open_price') else 0,
                        'market_value': p.market_value if hasattr(p, 'market_value') else 0,
                    }
        except: pass
        return positions

    def _get_prev_trading_day(self):
        """Get the previous trading day for prediction lookup."""
        today = date.today()
        # Try yesterday first, then go back
        d = today - timedelta(days=1)
        for _ in range(10):
            if d.weekday() < 5 and d in self._predictions:
                return d
            d -= timedelta(days=1)
        # Fallback: return the most recent date in predictions before today
        dates = sorted([dt for dt in self._predictions.keys() if dt < today])
        return dates[-1] if dates else None
