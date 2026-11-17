# -*- coding: utf-8 -*-
"""
Top10V8V1 — Hephaestus-Forge run_daily_pipeline_v8 每日Top10选股，固定持有2个交易日。

流程（14:50触发，每个交易日只执行一次）：
  1. 先处理到期批次卖出（买入后满2个交易日，14:50卖出）
  2. 调用 run_daily_pipeline_v8.py（不修改该脚本，独立子进程调用）拿到当日Top10
  3. 按“2批并存”方式计算单只权重并买入，记录本批次到期卖出日

仓位口径（用户确认）：同一时刻最多两批并存（今天新买 + 昨天未到期），
weight_per_stock = (1 - cash_buffer) / (top_n * hold_days)，
cash_buffer=0.05, top_n=10, hold_days=2 时约为总资金的4.75%/只，
稳态下最多同时持仓20只，总仓位不超100%。

回测对齐：label_horizon_days=2, entry_time="14:50"（不做止损/止盈/择时等额外风控叠加）。

Registration (conf/config.ini):
  [Top10V8V1]
  name = Top10V8V1
  path = strategy/top10_v8_v1.py
  config_file = conf/top10_v8_v1.ini
  interval = 60
"""

import csv
import json
import os
import logging
import asyncio
import subprocess
from datetime import datetime, date

import a_trade_calendar
from xtquant import xtconstant, xtdata
from base_strategy_v2 import BaseStrategy
from execution.order_protocol import AmbiguousSubmissionError, OrderIntent, OrderStatus

logger = logging.getLogger("Top10V8V1")


class Top10V8V1(BaseStrategy):
    """Hephaestus-Forge v8 Top10 每日选股，固定2交易日持有。"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        sec = "Top10V8V1"
        self.python_exe = self._cfg(sec, "python_exe", r"D:/tool/pyclient/python396/python.exe")
        self.forge_root = self._cfg(sec, "forge_root", r"D:/tool/pycharm_workspace/Hephaestus-Forge")
        self.top_n = int(self._cfg(sec, "top_n", "10"))
        self.hold_days = int(self._cfg(sec, "hold_days", "2"))
        self.cash_buffer = float(self._cfg(sec, "cash_buffer", "0.05"))
        self.rebalance_time = self._cfg(sec, "rebalance_time", "14:50")
        self.pipeline_timeout = int(self._cfg(sec, "pipeline_timeout", "1800"))
        self.is_test = self._cfg(sec, "is_test", "1")

        self.weight_per_stock = (1.0 - self.cash_buffer) / (self.top_n * self.hold_days)

        self._state_file = "D:/tool/dataset/strategy_data/top10_v8_v1_state.json"
        self.last_action_date = None
        self.batches = []  # [{buy_date, sell_date, positions: {code: {volume, name}}}]
        self._load_state()

        logger.info(
            f"[Top10V8V1] Init: top_n={self.top_n}, hold_days={self.hold_days}, "
            f"weight_per_stock={self.weight_per_stock:.2%}, is_test={self.is_test}, "
            f"batches_pending={len(self.batches)}"
        )

    # ========== Config helper ==========

    def _cfg(self, section, key, fallback):
        try:
            return self.config.get(section, key, fallback=fallback)
        except Exception:
            return fallback

    # ========== State persistence ==========

    def _load_state(self):
        try:
            if os.path.exists(self._state_file):
                with open(self._state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.last_action_date = state.get("last_action_date")
                self.batches = state.get("batches", [])
        except Exception as e:
            logger.warning(f"[Top10V8V1] State load failed: {e}")

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
            state = {
                "last_action_date": self.last_action_date,
                "batches": self.batches,
            }
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[Top10V8V1] State save failed: {e}")

    # ========== Main entry ==========

    async def do(self, accounts):
        req_dict = accounts.get("acc_1", {})
        xt_trader = req_dict.get("xt_trader")
        acc = req_dict.get("account")
        account_name = req_dict.get("acc_name", "未知")

        order_manager = req_dict.get("order_manager") if self.is_test != "1" else None

        if xt_trader is None or acc is None:
            return json.dumps({"warn": [{"mark": "no_account"}]})

        if self.is_test != "1" and (
            order_manager is None
            or not req_dict.get("managed_order_gate_open", False)
            or not getattr(order_manager, "submission_enabled", False)
        ):
            logger.error("[Top10V8V1] managed order gate is closed; reconciliation is required")
            return json.dumps({"warn": [{"mark": "order_reconciliation_required"}]})

        if order_manager is not None:
            self._sync_managed_order_fills(order_manager, account_name)

        cur_time = datetime.now().time()
        if not (datetime.strptime("09:30", "%H:%M").time() <= cur_time <= datetime.strptime("15:00", "%H:%M").time()):
            if self.is_test != "1":
                return json.dumps({"msg": "not_trade_time"})

        today = date.today()
        today_str = today.strftime("%Y-%m-%d")  # a_trade_calendar 要求字符串日期
        today_compact = today.strftime("%Y%m%d")

        if not a_trade_calendar.is_trade_date(today_str):
            return json.dumps({"msg": "not_trade_date"})

        if self.last_action_date == today_str:
            return json.dumps({"msg": "already_done_today"})

        rebalance_time = datetime.strptime(self.rebalance_time, "%H:%M").time()
        if cur_time < rebalance_time:
            return json.dumps({"msg": "before_rebalance_time"})

        # 1) 处理到期批次卖出
        self._process_sells(xt_trader, acc, account_name, today_str, order_manager=order_manager)

        # 2) 拉取今日Top10并买入新批次
        await self._process_buy(
            xt_trader, acc, account_name, today_str, today_compact,
            order_manager=order_manager,
        )

        self.last_action_date = today_str
        self._save_state()
        return json.dumps({"msg": "ok"})

    # ========== Sell (到期批次) ==========

    def _process_sells(self, xt_trader, acc, account_name, today_str, order_manager=None):
        remaining_batches = []
        for batch in self.batches:
            if batch.get("sell_date", "9999-99-99") > today_str:
                remaining_batches.append(batch)
                continue

            positions = batch.get("positions", {})
            codes = list(positions.keys())
            if not codes:
                continue

            try:
                ticks = xtdata.get_full_tick(codes)
            except Exception as e:
                logger.warning(f"[Top10V8V1] get_full_tick failed on sell: {e}")
                remaining_batches.append(batch)
                continue

            still_pending = {}
            for code, info in positions.items():
                tick = ticks.get(code)
                if tick is None:
                    still_pending[code] = info
                    continue
                try:
                    latest_price = float(tick["lastPrice"])
                    last_close = float(tick["lastClose"])
                except (KeyError, TypeError, ValueError):
                    still_pending[code] = info
                    continue
                if latest_price <= 0:
                    still_pending[code] = info
                    continue

                sell_price = round(latest_price * 0.99, 2)
                floor_price = round(last_close - last_close * 0.1, 2)
                if sell_price < floor_price:
                    sell_price = floor_price

                volume = info.get("volume", 0)
                if volume <= 0:
                    continue

                if order_manager is not None:
                    entry_client_order_id = info.get("entry_client_order_id")
                    entry_record = (
                        order_manager.store.get(entry_client_order_id)
                        if entry_client_order_id else None
                    )
                    if entry_record is None:
                        # A batch created by the legacy direct-QMT path has no
                        # durable entry id.  Selling it automatically could
                        # duplicate an outstanding legacy order.
                        logger.error(
                            "[Top10V8V1] legacy batch %s/%s needs manual migration",
                            batch.get("buy_date"), code,
                        )
                        still_pending[code] = info
                        continue
                    volume = int(entry_record.filled_quantity)
                    if volume <= 0:
                        still_pending[code] = info
                        continue

                    fixed_sell_price = float(info.get("sell_price", sell_price))
                    exit_client_order_id = info.get("exit_client_order_id")
                    existing_exit = (
                        order_manager.store.get(exit_client_order_id)
                        if exit_client_order_id else None
                    )
                    if existing_exit is not None:
                        if existing_exit.status in {
                            OrderStatus.PENDING,
                            OrderStatus.CANCEL_REQUESTED,
                            OrderStatus.UNKNOWN,
                        }:
                            still_pending[code] = info
                            continue
                        if existing_exit.status is OrderStatus.FILLED:
                            continue
                        if existing_exit.status is OrderStatus.PARTIALLY_FILLED:
                            volume -= int(existing_exit.filled_quantity)
                            if volume <= 0:
                                continue
                            info.pop("exit_client_order_id", None)
                            info["sell_signal_id"] = (
                                f"{today_str}:sell:{batch.get('buy_date', 'unknown')}:{code}:"
                                f"retry:{existing_exit.filled_quantity}"
                            )
                        else:
                            info.pop("exit_client_order_id", None)
                    signal_id = info.get(
                        "sell_signal_id",
                        f"{today_str}:sell:{batch.get('buy_date', 'unknown')}:{code}",
                    )
                    sell_intent = OrderIntent(
                        account_id=account_name,
                        strategy_name="Top10V8V1",
                        symbol=code,
                        side="SELL",
                        quantity=volume,
                        order_type="LIMIT",
                        limit_price=fixed_sell_price,
                        signal_id=signal_id,
                        prediction_date=today_str,
                    )
                    try:
                        sell_record = self._submit_managed_order(order_manager, sell_intent)
                    except Exception:
                        logger.exception("[Top10V8V1] managed sell rejected locally: %s", code)
                        still_pending[code] = info
                        continue
                    info["sell_price"] = fixed_sell_price
                    info["sell_signal_id"] = signal_id
                    info["exit_client_order_id"] = sell_record.client_order_id
                    if sell_record.status is not OrderStatus.FILLED:
                        still_pending[code] = info
                        logger.warning(
                            "[Top10V8V1] SELL %s remains %s client=%s",
                            code, sell_record.status.value, sell_record.client_order_id,
                        )
                    else:
                        logger.info(
                            "[Top10V8V1] SELL filled %s vol=%s avg=%.4f",
                            code, sell_record.filled_quantity, sell_record.avg_fill_price or 0,
                        )
                    continue

                order_id = -1
                if self.is_test != "1":
                    order_id = xt_trader.order_stock(
                        acc, code, xtconstant.STOCK_SELL, volume,
                        xtconstant.FIX_PRICE, sell_price, strategy_name="top10_v8"
                    )
                self.record_trade(
                    action="sell", stock_code=code, stock_name=info.get("name", "未知"),
                    price=sell_price, volume=volume, order_id=order_id,
                    reason=f"持有满{self.hold_days}个交易日到期卖出", account=account_name
                )
                logger.info(f"[Top10V8V1] SELL {code} vol={volume} price={sell_price}")

            if still_pending:
                batch["positions"] = still_pending
                remaining_batches.append(batch)

        self.batches = remaining_batches

    # ========== Managed order bridge ==========

    @staticmethod
    def _submit_managed_order(order_manager, intent):
        """Submit one durable intent and fail closed on ambiguous outcomes."""
        record = order_manager.submit(intent)
        if record.status in {OrderStatus.UNKNOWN, OrderStatus.REJECTED}:
            raise AmbiguousSubmissionError(
                f"managed order {record.client_order_id} is {record.status.value}"
            )
        return record

    def _sync_managed_order_fills(self, order_manager, account_name):
        """Apply each newly observed fill to strategy holdings exactly once."""
        for batch in self.batches:
            for code, info in batch.get("positions", {}).items():
                entry_id = info.get("entry_client_order_id")
                entry = order_manager.store.get(entry_id) if entry_id else None
                if entry is not None:
                    if entry.status is OrderStatus.REJECTED:
                        info["volume"] = 0
                        continue
                    filled = int(entry.filled_quantity)
                    previous = int(info.get("entry_recorded_filled", 0))
                    if filled > previous:
                        self.record_trade(
                            action="buy",
                            stock_code=code,
                            stock_name=info.get("name", "未知"),
                            price=float(entry.avg_fill_price or info.get("buy_price", 0.0)),
                            volume=filled - previous,
                            order_id=entry.broker_order_id or -1,
                            reason="managed_order_fill",
                            account=account_name,
                        )
                        info["entry_recorded_filled"] = filled
                    info["volume"] = filled

                exit_id = info.get("exit_client_order_id")
                exit_record = order_manager.store.get(exit_id) if exit_id else None
                if exit_record is None:
                    continue
                filled = int(exit_record.filled_quantity)
                previous = int(info.get("exit_recorded_filled", 0))
                if filled > previous:
                    self.record_trade(
                        action="sell",
                        stock_code=code,
                        stock_name=info.get("name", "未知"),
                        price=float(exit_record.avg_fill_price or info.get("sell_price", 0.0)),
                        volume=filled - previous,
                        order_id=exit_record.broker_order_id or -1,
                        reason="managed_order_fill",
                        account=account_name,
                    )
                    info["exit_recorded_filled"] = filled


    # ========== Buy (今日新批次) ==========

    async def _process_buy(
        self, xt_trader, acc, account_name, today_str, today_compact,
        order_manager=None,
    ):
        picks = await self._run_pipeline_and_load_picks(today_compact)
        if not picks:
            logger.warning(f"[Top10V8V1] No picks for {today_compact}, skip buy")
            return

        pv = self._get_portfolio_value(xt_trader, acc)
        cash = self._get_cash(xt_trader, acc)
        if pv <= 0:
            logger.warning("[Top10V8V1] portfolio value <= 0, skip buy")
            return

        pending_codes = self._get_pending_order_codes(xt_trader, acc)

        codes = [p["stock_code"] for p in picks]
        try:
            ticks = xtdata.get_full_tick(codes)
        except Exception as e:
            logger.warning(f"[Top10V8V1] get_full_tick failed on buy: {e}")
            return

        lot = 100
        target_value = self.weight_per_stock * pv
        new_positions = {}

        for rank, p in enumerate(picks, start=1):
            code = p["stock_code"]
            name = p.get("stock_name", "未知")

            if code in pending_codes:
                continue

            tick = ticks.get(code)
            if tick is None:
                continue
            try:
                price = float(tick["lastPrice"])
            except (KeyError, TypeError, ValueError):
                continue
            if price <= 0:
                continue

            qty = int(target_value / price)
            qty = (qty // lot) * lot
            if qty <= 0:
                continue

            buy_price = round(price * 1.005, 2)
            cost = qty * buy_price
            if cost > cash:
                qty = int(cash / buy_price) // lot * lot
                if qty <= 0:
                    continue
                cost = qty * buy_price

            order_id = -1
            managed_record = None
            if order_manager is not None:
                buy_intent = OrderIntent(
                    account_id=account_name,
                    strategy_name="Top10V8V1",
                    symbol=code,
                    side="BUY",
                    quantity=qty,
                    order_type="LIMIT",
                    limit_price=buy_price,
                    signal_id=f"{today_compact}:buy:{rank}:{code}",
                    prediction_date=today_str,
                )
                try:
                    managed_record = self._submit_managed_order(order_manager, buy_intent)
                    order_id = managed_record.broker_order_id or -1
                except Exception:
                    logger.exception("[Top10V8V1] managed buy rejected locally: %s", code)
                    continue
            elif self.is_test != "1":
                order_id = xt_trader.order_stock(
                    acc, code, xtconstant.STOCK_BUY, qty,
                    xtconstant.FIX_PRICE, buy_price, strategy_name="top10_v8"
                )
            if managed_record is not None:
                logger.info(
                    "[Top10V8V1] managed BUY %s vol=%s price=%s status=%s client=%s broker=%s",
                    code, qty, buy_price, managed_record.status.value,
                    managed_record.client_order_id, managed_record.broker_order_id,
                )
                cash -= cost
                new_positions[code] = {
                    "volume": qty,
                    "name": name,
                    "entry_client_order_id": managed_record.client_order_id,
                    "buy_price": buy_price,
                }
                continue
            self.record_trade(
                action="buy", stock_code=code, stock_name=name,
                price=buy_price, volume=qty, order_id=order_id,
                reason="v8模型Top10选股买入", account=account_name
            )
            logger.info(f"[Top10V8V1] BUY {code}({name}) vol={qty} price={buy_price}")

            cash -= cost
            new_positions[code] = {"volume": qty, "name": name}

        if new_positions:
            sell_date = a_trade_calendar.get_next_trade_date(today_str, self.hold_days)
            self.batches.append({
                "buy_date": today_str,
                "sell_date": sell_date,
                "positions": new_positions,
            })
            logger.info(f"[Top10V8V1] New batch: buy={today_str}, sell={sell_date}, n={len(new_positions)}")

    # ========== Pipeline invocation (non-blocking) ==========

    async def _run_pipeline_and_load_picks(self, today_compact):
        cmd = [
            self.python_exe, "-u", "-m", "top10_ranker.export.run_daily_pipeline_v8",
            today_compact, "--top-k", str(self.top_n),
        ]
        try:
            result = await asyncio.to_thread(
                subprocess.run, cmd, cwd=self.forge_root,
                capture_output=True, text=True, timeout=self.pipeline_timeout,
            )
        except Exception as e:
            logger.error(f"[Top10V8V1] pipeline subprocess failed: {e}")
            return []

        if result.returncode != 0:
            logger.error(
                f"[Top10V8V1] pipeline exit={result.returncode}, "
                f"stderr(tail)={result.stderr[-2000:] if result.stderr else ''}"
            )
            return []

        csv_path = os.path.join(
            self.forge_root, "top10_ranker", "output", f"top10_{today_compact}_v8.csv"
        )
        if not os.path.exists(csv_path):
            logger.error(f"[Top10V8V1] pipeline output not found: {csv_path}")
            return []

        picks = []
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    picks.append({
                        "rank": int(row["rank"]),
                        "stock_code": row["stock_code"].strip(),
                        "stock_name": row["stock_name"].strip(),
                        "score": float(row["score"]),
                    })
        except Exception as e:
            logger.error(f"[Top10V8V1] failed to parse {csv_path}: {e}")
            return []

        picks.sort(key=lambda x: x["rank"])
        return picks[: self.top_n]

    # ========== Helpers ==========

    def _get_portfolio_value(self, xt_trader, acc):
        try:
            pos_val = sum(
                p.market_value for p in xt_trader.query_stock_positions(acc)
                if hasattr(p, "market_value") and p.market_value
            )
            asset = xt_trader.query_stock_asset(acc)
            return (asset.cash if asset else 0) + pos_val
        except Exception:
            return 0.0

    def _get_cash(self, xt_trader, acc):
        try:
            asset = xt_trader.query_stock_asset(acc)
            return asset.cash if asset else 0.0
        except Exception:
            return 0.0

    def _get_pending_order_codes(self, xt_trader, acc):
        codes = set()
        try:
            for wt in xt_trader.query_stock_orders(acc, True):
                if wt.stock_code:
                    codes.add(wt.stock_code)
        except Exception:
            pass
        return codes
