"""Compatibility facade that routes legacy QMT submissions through OrderManager.

Strategies can keep using the query APIs exposed by ``XtQuantTrader`` while
every order submission is converted into the durable broker-neutral protocol.
This is the migration boundary for strategies that have not yet been edited to
construct ``OrderIntent`` themselves.
"""

from __future__ import annotations

import datetime
import inspect
import logging
from types import SimpleNamespace
from typing import Any

from .order_protocol import AmbiguousSubmissionError, ExecutionGateClosed, OrderIntent, OrderStatus

logger = logging.getLogger(__name__)


class ManagedTrader:
    """Delegate reads to QMT and make ``order_stock`` durable and idempotent."""

    def __init__(self, trader: Any, account: Any, manager: Any, *, gate_open: bool,
                 account_id: str = "", execution_channel: str = "mini_qmt",
                 account_gateway: Any = None) -> None:
        self._trader = trader
        self._account = account
        self._manager = manager
        self._gate_open = bool(gate_open)
        self._account_id = str(account_id or "")
        self._execution_channel = str(execution_channel or "mini_qmt").lower()
        self._account_gateway = account_gateway

    @property
    def gate_open(self) -> bool:
        """Current submission gate; can be closed by connection supervisors."""
        return self._gate_open and bool(
            self._manager is not None
            and getattr(self._manager, "submission_enabled", True)
        )

    def set_gate(self, open_: bool) -> None:
        self._gate_open = bool(open_)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._trader, name)

    def query_stock_asset(self, account):
        if self._account_gateway is None:
            return self._trader.query_stock_asset(account)
        snapshot = self._account_gateway.query_account(self._account_id or str(account))
        return SimpleNamespace(
            cash=float(snapshot.cash),
            available_cash=float(snapshot.available_cash),
            total_asset=float(snapshot.equity),
            market_value=sum(float(position.market_value) for position in snapshot.positions),
        )

    def query_stock_positions(self, account):
        if self._account_gateway is None:
            return self._trader.query_stock_positions(account)
        positions = self._account_gateway.query_positions(self._account_id or str(account))
        return [SimpleNamespace(
            stock_code=position.symbol,
            volume=position.quantity,
            can_use_volume=position.available_quantity,
            open_price=position.average_price,
            market_value=position.market_value,
            side=position.side,
        ) for position in positions]

    def query_stock_orders(self, account, cancelable_only=False):
        if self._account_gateway is None:
            return self._trader.query_stock_orders(account, cancelable_only)
        query = getattr(self._account_gateway, "query_orders", None)
        if query is None:
            return []
        return [SimpleNamespace(
            order_id=snapshot.broker_order_id,
            order_sysid=snapshot.broker_order_sys_id,
            order_remark=snapshot.client_order_id,
            stock_code=snapshot.symbol,
            order_status=snapshot.status,
            traded_volume=snapshot.filled_quantity,
            traded_price=snapshot.avg_fill_price or 0,
            status_msg=snapshot.reject_reason,
        ) for snapshot in query(cancelable_only=cancelable_only)]

    def order_stock(self, account, symbol, order_type, quantity, price_type, price,
                    strategy_name=None, order_remark=None, **kwargs):
        if not self.gate_open or self._manager is None:
            logger.error("managed order gate is closed; rejecting legacy submission for %s", symbol)
            return -1
        buy_value = getattr(self._constant_module(), "STOCK_BUY", 23)
        side = "BUY" if int(order_type) == int(buy_value) else "SELL"
        latest_value = getattr(self._constant_module(), "LATEST_PRICE", 5)
        order_kind = "MARKET" if int(price_type) == int(latest_value) else "LIMIT"
        caller = inspect.stack()[1].function
        strategy = str(strategy_name or "legacy")
        signal_id = str(order_remark or "{}:{}:{}:{}:{}:{}".format(
            datetime.date.today().isoformat(), strategy, caller, symbol, side, int(quantity)
        ))
        intent = OrderIntent(
            account_id=self._account_id or str(getattr(account, "account_id", self._account)),
            strategy_name=strategy,
            symbol=str(symbol), side=side, quantity=int(quantity),
            order_type=order_kind, limit_price=None if order_kind == "MARKET" else float(price),
            signal_id=signal_id,
            # Legacy callers do not provide a stable signal timestamp.  Keep
            # it empty so repeated calls for the same logical signal reuse the
            # deterministic client_order_id instead of creating duplicates.
            signal_timestamp=None,
            metadata={
                "legacy_submission": True,
                "caller": caller,
                "execution_channel": self._execution_channel,
            },
        )
        try:
            record = self._manager.submit(intent)
        except ExecutionGateClosed:
            logger.warning("managed order gate is closed; rejecting %s", intent.client_order_id)
            return -1
        except AmbiguousSubmissionError:
            logger.exception("ambiguous managed submission for %s", intent.client_order_id)
            return -1
        if record.status in {OrderStatus.UNKNOWN, OrderStatus.REJECTED}:
            logger.error("managed submission %s is %s", record.client_order_id, record.status.value)
            return -1
        return int(record.broker_order_id) if record.broker_order_id else -1

    @staticmethod
    def _constant_module():
        try:
            from xtquant import xtconstant
            return xtconstant
        except Exception:
            return None
