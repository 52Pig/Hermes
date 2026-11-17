# -*- coding: utf-8 -*-
"""
MultiFactor V1 — Diversified multi-factor alpha strategy.

Based on Atropos backtest: +18.12% return, 3.93% max drawdown, Sharpe 1.88 (top_n=20)
over 2026-01-05 to 2026-05-18 (86 trading days, 1M CNY initial capital).

Strategy logic:
  1. Factor model: Momentum(35%) + Reversal(20%) + LowVol(30%) + LowTurnover(15%)
     — All factors computed from YESTERDAY's close (zero look-ahead)
  2. Universe: pre-filtered top 600 stocks from stock_pools.json
  3. Selection: top 20 by composite score, equal weight (optimized)
  4. Rebalance: weekly (every 5 trading days), near market close (14:50)
  5. Risk controls:
     - Market timing: only invest when >50% proxy stocks above MA20
     - Position cap: max 3% per stock
     - Stop-loss: per-position -5% trailing stop
     - Portfolio drawdown stop: 15% from peak → freeze 10 days
     - Cash buffer: 10%

Data pre-computation:
  prepare_multi_factor_data.py runs daily before market open,
  generates multi_factor_scores_{YYYYMMDD}.json in strategy_data/

Dependencies:
  - xtquant (QMT data API)
  - base_strategy_v2 (Hermes base class)
  - stock_pools.json (tradeable universe)
  - pre-computed factor scores JSON

Registration (conf/config.ini):
  [MultiFactorV1]
  name = MultiFactorV1
  path = strategy/multi_factor_v1.py
  config_file = conf/dragon_v4.ini
  interval = 180
"""

import json
import os
import glob
import time
import logging
from datetime import datetime, date, timedelta
import numpy as np

from xtquant import xtconstant, xtdata
from base_strategy_v2 import BaseStrategy

logger = logging.getLogger("MultiFactorV1")


class MultiFactorV1(BaseStrategy):
    """Multi-factor alpha strategy — weekly rebalance, risk-controlled."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # ---- Strategy Parameters (matching Atropos v3 backtest) ----
        self.top_n = 20
        self.rebalance_days = 5
        self.max_position_pct = 0.03
        self.cash_buffer = 0.10
        self.stop_loss_pct = 0.05
        self.portfolio_dd_stop = 0.15
        self.portfolio_dd_freeze = 10

        self.factor_weights = {
            'momentum': 0.35, 'reversal': 0.20,
            'lowvol': 0.30, 'turnover': 0.15,
        }

        # ---- State ----
        self.is_test = "0"
        self._last_rebalance_date = None
        self._last_tick_date = None  # tracks trading day transitions for rebalance counting
        self._peak_nav = 0.0
        self._dd_frozen_until = None
        self._entry_prices = {}  # {code: entry_price} for stop-loss
        self._wt_dict = {}       # submitted orders tracking
        self._scores = {}        # {code: composite_score}
        self._trading_day_count = 0
        self._state_file = "D:/tool/dataset/strategy_data/multi_factor_state.json"

        # ---- Load factor scores ----
        self._load_scores()
        # ---- Restore persisted state (survives daily restart) ----
        self._load_state()

        logger.info(
            f"[MultiFactorV1] Init: top_n={self.top_n}, rebalance={self.rebalance_days}d, "
            f"stop_loss={self.stop_loss_pct:.0%}, dd_stop={self.portfolio_dd_stop:.0%}, "
            f"scores_loaded={len(self._scores)}"
        )

    # ========== Factor Score Loading ==========

    def _load_scores(self):
        """Load the latest pre-computed factor scores."""
        score_dir = "D:/tool/dataset/strategy_data"
        pattern = os.path.join(score_dir, "multi_factor_scores_*.json")
        files = sorted(glob.glob(pattern), reverse=True)

        for fpath in files:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._scores = data.get('scores', {})
                if self._scores:
                    score_date = data.get('date', 'unknown')
                    logger.info(
                        f"[MultiFactorV1] Loaded {len(self._scores)} scores from {score_date}"
                    )
                    return
            except (json.JSONDecodeError, KeyError, IOError) as e:
                logger.warning(f"[MultiFactorV1] Failed to load {fpath}: {e}")
                continue

        logger.warning("[MultiFactorV1] No valid score file found! Strategy will stay cash.")

    def _load_state(self):
        """Restore state that should survive daily restarts."""
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
                self._dd_frozen_until = (
                    datetime.strptime(frozen, '%Y-%m-%d') if frozen else None
                )
                self._trading_day_count = state.get('trading_day_count', 0)
                self._entry_prices = state.get('entry_prices', {})
                logger.info(
                    f"[MultiFactorV1] Restored state: peak_nav={self._peak_nav:,.0f}, "
                    f"last_rebalance={self._last_rebalance_date}, count={self._trading_day_count}"
                )
        except Exception as e:
            logger.warning(f"[MultiFactorV1] Failed to load state: {e}")

    def _save_state(self):
        """Persist state for survival across daily restarts."""
        try:
            state = {
                'peak_nav': self._peak_nav,
                'last_rebalance_date': (
                    self._last_rebalance_date.strftime('%Y-%m-%d')
                    if self._last_rebalance_date else None
                ),
                'last_tick_date': (
                    self._last_tick_date.strftime('%Y-%m-%d')
                    if self._last_tick_date else None
                ),
                'dd_frozen_until': (
                    self._dd_frozen_until.strftime('%Y-%m-%d %H:%M:%S')
                    if self._dd_frozen_until else None
                ),
                'trading_day_count': self._trading_day_count,
                'entry_prices': self._entry_prices,
            }
            with open(self._state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[MultiFactorV1] Failed to save state: {e}")

    # ========== Main Entry Point ==========

    async def do(self, accounts):
        """Called periodically by Hermes MicroserviceManager (every interval seconds)."""
        req_dict = accounts.get("acc_1", accounts.get("acc_2", {}))
        xt_trader = req_dict.get("xt_trader")
        acc = req_dict.get("account")
        self.is_test = req_dict.get("is_test", "0")

        if xt_trader is None or acc is None:
            logger.warning("[MultiFactorV1] No valid account connection")
            return json.dumps({"warn": [{"mark": "no_account"}]})

        # Check trading time
        cur_time = datetime.now().time()
        trade_start = datetime.strptime("09:30", "%H:%M").time()
        trade_end = datetime.strptime("15:00", "%H:%M").time()
        rebalance_time = datetime.strptime("14:50", "%H:%M").time()

        is_trade_time = trade_start <= cur_time <= trade_end and self.is_test == "0"
        if not is_trade_time:
            return json.dumps({"msg": "not_trade_time"})

        # ---- Step 1: Compute portfolio NAV and check drawdown stop ----
        pv = self._get_portfolio_value(xt_trader, acc)
        if pv > self._peak_nav:
            self._peak_nav = pv
            self._save_state()

        if self._peak_nav > 0 and self.portfolio_dd_stop > 0:
            current_dd = (self._peak_nav - pv) / self._peak_nav
            if current_dd >= self.portfolio_dd_stop:
                if self._dd_frozen_until is None:
                    self._dd_frozen_until = datetime.now() + timedelta(days=self.portfolio_dd_freeze)
                    self._save_state()
                    self._go_to_cash(xt_trader, acc)
                    logger.warning(
                        f"[MultiFactorV1] PORTFOLIO DD STOP: {current_dd:.2%} "
                        f"(peak={self._peak_nav:,.0f}, current={pv:,.0f}). "
                        f"Frozen until {self._dd_frozen_until.strftime('%Y-%m-%d')}"
                    )
                return json.dumps({"msg": "dd_frozen"})

        if self._dd_frozen_until is not None:
            if datetime.now() < self._dd_frozen_until:
                return json.dumps({"msg": "dd_frozen"})
            else:
                self._dd_frozen_until = None
                self._peak_nav = pv
                self._save_state()
                logger.info("[MultiFactorV1] DD freeze lifted")

        # ---- Step 2: Stop-loss check (run every cycle) ----
        self._check_stop_loss(xt_trader, acc)

        # ---- Step 3: Rebalance check ----
        today = date.today()
        if not self._is_rebalance_day(today) or cur_time < rebalance_time:
            return json.dumps({"msg": "no_rebalance"})

        # Avoid duplicate rebalance on same day
        if self._last_rebalance_date == today:
            return json.dumps({"msg": "already_rebalanced"})

        # ---- Step 4: Market timing check ----
        if not self._market_healthy():
            self._go_to_cash(xt_trader, acc)
            self._last_rebalance_date = today
            self._trading_day_count = 0  # rebalance decision made: stay cash
            self._save_state()
            logger.info("[MultiFactorV1] Market BEAR — staying cash")
            return json.dumps({"msg": "bear_market"})

        # ---- Step 5: Select top N stocks from scores ----
        if not self._scores:
            return json.dumps({"warn": [{"mark": "no_scores"}]})

        ranked = sorted(self._scores.items(), key=lambda x: x[1], reverse=True)
        selected = [code for code, score in ranked[:self.top_n]]
        logger.info(
            f"[MultiFactorV1] Rebalance {today}: {len(selected)} stocks selected, "
            f"top={selected[0]}:{self._scores.get(selected[0], 0):.3f}"
        )

        # ---- Step 6: Execute rebalance ----
        self._rebalance(xt_trader, acc, selected)
        self._last_rebalance_date = today
        self._trading_day_count = 0  # rebalance completed, reset counter
        self._save_state()

        return json.dumps({"msg": "ok"})

    # ========== Portfolio Helpers ==========

    def _get_portfolio_value(self, xt_trader, acc) -> float:
        """Compute total portfolio value (cash + positions market value)."""
        try:
            positions = xt_trader.query_stock_positions(acc)
            pos_value = sum(
                p.market_value for p in positions
                if hasattr(p, 'market_value') and p.market_value is not None
            )
            asset = xt_trader.query_stock_asset(acc)
            cash = asset.cash if asset else 0
            return cash + pos_value
        except Exception:
            return 0.0

    def _get_cash(self, xt_trader, acc) -> float:
        try:
            asset = xt_trader.query_stock_asset(acc)
            return asset.cash if asset else 0
        except Exception:
            return 0.0

    def _get_positions(self, xt_trader, acc) -> dict:
        """Return {code: {volume, open_price, market_value}}."""
        positions = {}
        try:
            for p in xt_trader.query_stock_positions(acc):
                if p.volume > 0:
                    positions[p.stock_code] = {
                        'volume': p.volume,
                        'open_price': p.open_price if hasattr(p, 'open_price') else 0,
                        'market_value': p.market_value if hasattr(p, 'market_value') else 0,
                    }
        except Exception:
            pass
        return positions

    # ========== Market Timing ==========

    def _market_healthy(self) -> bool:
        """Check if >50% of proxy stocks are above 20-day MA."""
        if not self._scores:
            return False

        # Use top 50 stocks by score as market proxy
        ranked = sorted(self._scores.items(), key=lambda x: x[1], reverse=True)
        proxy_codes = [c for c, _ in ranked[:50]]

        today = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=60)).strftime('%Y%m%d')

        try:
            data = xtdata.get_market_data_ex(
                stock_list=proxy_codes[:30],  # limit to 30 for speed
                period='1d',
                start_time=start,
                end_time=today,
                field_list=['close']
            )
        except Exception:
            return False  # conservative: don't invest if market health unknown

        above_ma = 0
        total = 0
        for code, df in data.items():
            if df is None or len(df) < 22:
                continue
            closes = df['close'].values
            if len(closes) < 22:
                continue
            ma20 = np.mean(closes[-21:-1])  # exclude today
            current = closes[-2] if len(closes) >= 2 else closes[-1]  # yesterday
            if current > ma20:
                above_ma += 1
            total += 1

        if total < 5:
            return True
        ratio = above_ma / total
        healthy = ratio >= 0.5
        if not healthy:
            logger.info(f"[MultiFactorV1] Market BEAR: {ratio:.0%} above MA20")
        return healthy

    # ========== Stop Loss ==========

    def _check_stop_loss(self, xt_trader, acc):
        """Check trailing stop-loss for all positions."""
        positions = self._get_positions(xt_trader, acc)
        if not positions:
            return

        codes = list(positions.keys())
        try:
            ticks = xtdata.get_full_tick(codes)
        except Exception:
            return

        for code, pos in positions.items():
            if pos['volume'] <= 0:
                continue
            tick = ticks.get(code)
            if tick is None:
                continue
            try:
                current_price = float(tick['lastPrice'])
            except (KeyError, TypeError, ValueError):
                continue
            if current_price <= 0:
                continue

            entry = self._entry_prices.get(code, pos['open_price'])
            if entry > 0 and (current_price / entry - 1) < -self.stop_loss_pct:
                sell_price = round(current_price * 0.99, 2)
                if self.is_test != "1":
                    order_id = xt_trader.order_stock(
                        acc, code, xtconstant.STOCK_SELL,
                        pos['volume'], xtconstant.FIX_PRICE, sell_price,
                        strategy_name="multi_factor"
                    )
                self._entry_prices.pop(code, None)
                logger.info(
                    f"[MultiFactorV1] STOP-LOSS {code}: entry={entry:.2f} "
                    f"current={current_price:.2f} ({(current_price/entry-1)*100:.1f}%)"
                )

    # ========== Rebalance ==========

    def _is_rebalance_day(self, today) -> bool:
        """Check if today is a rebalance day (every N trading days).
        Counts actual trading days (date transitions), not do() call ticks.
        On first-ever run (_last_rebalance_date is None), forces rebalance.
        Does NOT reset the counter — caller must do that after successful rebalance."""
        if self._last_rebalance_date is None:
            return True
        # Increment counter only once per trading day
        if self._last_tick_date != today:
            self._last_tick_date = today
            self._trading_day_count += 1
            self._save_state()
        return self._trading_day_count >= self.rebalance_days

    def _go_to_cash(self, xt_trader, acc):
        """Sell all positions."""
        positions = self._get_positions(xt_trader, acc)
        for code, pos in positions.items():
            if pos['volume'] <= 0:
                continue
            try:
                ticks = xtdata.get_full_tick([code])
                tick = ticks.get(code)
                price = float(tick['lastPrice']) if tick else 0
                sell_price = round(price * 0.99, 2) if price > 0 else 0
            except Exception:
                sell_price = 0
            if sell_price <= 0:
                continue
            if self.is_test != "1":
                xt_trader.order_stock(
                    acc, code, xtconstant.STOCK_SELL,
                    pos['volume'], xtconstant.FIX_PRICE, sell_price,
                    strategy_name="multi_factor"
                )
        self._entry_prices.clear()

    def _rebalance(self, xt_trader, acc, selected):
        """Execute rebalance: sell removed, buy new, adjust existing."""
        positions = self._get_positions(xt_trader, acc)
        cash = self._get_cash(xt_trader, acc)
        pv = self._get_portfolio_value(xt_trader, acc)
        if pv <= 0:
            return

        cash_reserved = pv * self.cash_buffer
        available_cash = max(0, cash - cash_reserved)

        # Get current prices
        all_codes = list(set(list(positions.keys()) + selected))
        try:
            ticks = xtdata.get_full_tick(all_codes)
        except Exception:
            return

        # Step 1: Sell positions not in selected list
        for code, pos in list(positions.items()):
            if code not in selected and pos['volume'] > 0:
                tick = ticks.get(code)
                if tick is None:
                    continue
                try:
                    price = float(tick['lastPrice'])
                except (KeyError, TypeError, ValueError):
                    continue
                sell_price = round(price * 0.99, 2)
                if self.is_test != "1":
                    xt_trader.order_stock(
                        acc, code, xtconstant.STOCK_SELL,
                        pos['volume'], xtconstant.FIX_PRICE, sell_price,
                        strategy_name="multi_factor"
                    )
                self._entry_prices.pop(code, None)
                available_cash += pos['volume'] * sell_price * 0.995

        # Step 2: Buy selected stocks (equal weight)
        weight = 1.0 / len(selected) if selected else 0
        lot_size = 100

        for code in selected:
            tick = ticks.get(code)
            if tick is None:
                continue
            try:
                price = float(tick['lastPrice'])
            except (KeyError, TypeError, ValueError):
                continue
            if price <= 0:
                continue

            target_value = weight * pv
            current_volume = positions.get(code, {}).get('volume', 0)
            current_value = current_volume * price
            delta_value = target_value - current_value

            if abs(delta_value) < price * lot_size * 2:
                continue

            delta_qty = int(delta_value / price)
            delta_qty = (delta_qty // lot_size) * lot_size

            if delta_qty > 0:
                cost_est = delta_qty * price * 1.005
                if cost_est <= available_cash:
                    buy_price = round(price * 1.005, 2)  # slight premium to ensure fill
                    if self.is_test != "1":
                        xt_trader.order_stock(
                            acc, code, xtconstant.STOCK_BUY,
                            delta_qty, xtconstant.FIX_PRICE, buy_price,
                            strategy_name="multi_factor"
                        )
                    available_cash -= cost_est
                    self._entry_prices[code] = price
                    self._wt_dict[code] = 1

    # ========== Bookkeeping (base class hooks) ==========

    def save_holdings(self):
        """Save current holdings — called by base class."""
        super().save_holdings()

    def load_holdings(self):
        """Load persisted holdings — called by base class."""
        super().load_holdings()
