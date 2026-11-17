"""QMT/xtquant adapter for the broker-neutral order protocol."""

from __future__ import annotations

import logging
import threading
from typing import Any, Iterable, Optional

from .order_protocol import (
    BrokerOrderSnapshot,
    FillEvent,
    OrderManager,
    OrderRecord,
    OrderSide,
    OrderStatus,
    OrderType,
    SubmissionResult,
)
from .account_protocol import AccountSnapshot, PositionSnapshot

logger = logging.getLogger(__name__)

try:  # xtquant is installed only in a QMT runtime, not in offline tests.
    from xtquant import xtconstant  # type: ignore
    from xtquant.xttrader import XtQuantTraderCallback  # type: ignore
except Exception:  # pragma: no cover - exercised by offline environments
    xtconstant = None

    class XtQuantTraderCallback:  # type: ignore[no-redef]
        pass


def _constant(name: str, fallback: int) -> int:
    return int(getattr(xtconstant, name, fallback)) if xtconstant is not None else fallback


def _order_status(value: Any) -> OrderStatus:
    if isinstance(value, OrderStatus):
        return value
    numeric = {
        _constant("ORDER_UNREPORTED", 48): OrderStatus.PENDING,
        _constant("ORDER_WAIT_REPORTING", 49): OrderStatus.PENDING,
        _constant("ORDER_REPORTED", 50): OrderStatus.PENDING,
        _constant("ORDER_REPORTED_CANCEL", 51): OrderStatus.CANCEL_REQUESTED,
        _constant("ORDER_PARTSUCC_CANCEL", 52): OrderStatus.CANCEL_REQUESTED,
        _constant("ORDER_PART_CANCEL", 53): OrderStatus.CANCELLED,
        _constant("ORDER_CANCELED", 54): OrderStatus.CANCELLED,
        _constant("ORDER_PART_SUCC", 55): OrderStatus.PARTIALLY_FILLED,
        _constant("ORDER_SUCCEEDED", 56): OrderStatus.FILLED,
        _constant("ORDER_JUNK", 57): OrderStatus.REJECTED,
        _constant("ORDER_UNKNOWN", 255): OrderStatus.UNKNOWN,
    }
    return numeric.get(value, OrderStatus.UNKNOWN)


class XtQuantGateway:
    """Synchronous adapter around ``XtQuantTrader``.

    ``client_order_id`` is sent as QMT's ``order_remark``.  That is the key
    used by restart reconciliation; a gateway that cannot preserve the remark
    must not be used for managed orders.
    """

    def __init__(self, xt_trader: Any, account: Any, *, strategy_prefix: str = "hermes") -> None:
        self.xt_trader = xt_trader
        self.account = account
        self.strategy_prefix = strategy_prefix

    def submit(self, intent) -> SubmissionResult:
        order_type = _constant("STOCK_BUY", 23) if intent.side == OrderSide.BUY.value else _constant("STOCK_SELL", 24)
        price_type = _constant("FIX_PRICE", 11) if intent.order_type == OrderType.LIMIT.value else _constant("LATEST_PRICE", 5)
        strategy_name = f"{self.strategy_prefix}:{intent.strategy_name}"
        broker_order_id = self.xt_trader.order_stock(
            self.account,
            intent.symbol,
            order_type,
            int(intent.quantity),
            price_type,
            float(intent.limit_price or 0.0),
            strategy_name=strategy_name,
            order_remark=intent.client_order_id,
        )
        accepted = broker_order_id is not None and int(broker_order_id) >= 0
        return SubmissionResult(
            accepted=accepted,
            broker_order_id=str(broker_order_id) if accepted else None,
            reason=None if accepted else f"qmt_order_stock_returned_{broker_order_id}",
        )

    def cancel(self, record: OrderRecord) -> bool:
        if not record.broker_order_id:
            return False
        result = self.xt_trader.cancel_order_stock(self.account, int(record.broker_order_id))
        return int(result) == 0

    def query_account(self, account_id: str) -> AccountSnapshot:
        asset = self.xt_trader.query_stock_asset(self.account)
        positions = tuple(self.query_positions(account_id))
        cash = float(getattr(asset, "cash", 0) or 0) if asset else 0.0
        available = float(getattr(asset, "cash", 0) or 0) if asset else 0.0
        equity = float(getattr(asset, "total_asset", 0) or 0) if asset else 0.0
        if equity <= 0:
            equity = cash + sum(position.market_value for position in positions)
        return AccountSnapshot(
            account_id=str(account_id), currency="CNY", cash=cash,
            available_cash=available, equity=equity, positions=positions,
            metadata={"channel": "mini_qmt"},
        )

    def query_positions(self, account_id: str) -> Iterable[PositionSnapshot]:
        for position in self.xt_trader.query_stock_positions(self.account) or []:
            quantity = float(getattr(position, "volume", 0) or 0)
            if quantity <= 0:
                continue
            yield PositionSnapshot(
                account_id=str(account_id),
                symbol=str(getattr(position, "stock_code", "")),
                quantity=quantity,
                available_quantity=float(getattr(position, "can_use_volume", quantity) or 0),
                average_price=float(getattr(position, "open_price", 0) or 0),
                market_value=float(getattr(position, "market_value", 0) or 0),
                side="LONG",
                metadata={"channel": "mini_qmt"},
            )

    def query_orders(self, *, cancelable_only: bool = False) -> Iterable[BrokerOrderSnapshot]:
        for order in self.xt_trader.query_stock_orders(self.account, cancelable_only) or []:
            yield BrokerOrderSnapshot(
                broker_order_id=str(getattr(order, "order_id", "")) or None,
                broker_order_sys_id=str(getattr(order, "order_sysid", "")) or None,
                client_order_id=getattr(order, "order_remark", "") or None,
                symbol=str(getattr(order, "stock_code", "")),
                status=_order_status(getattr(order, "order_status", 255)),
                filled_quantity=int(getattr(order, "traded_volume", 0) or 0),
                avg_fill_price=float(getattr(order, "traded_price", 0) or 0) or None,
                reject_reason=getattr(order, "status_msg", "") or None,
            )


class XtQuantCallbackBridge(XtQuantTraderCallback):
    """Translate QMT callbacks into idempotent protocol events."""

    def __init__(self, manager: OrderManager) -> None:
        self.manager = manager
        self._connected = False
        self._connection_lock = threading.RLock()
        self._connection_revision = 0
        self.last_disconnect_reason = None
        self._gate_controllers = []

    @property
    def connected(self) -> bool:
        with self._connection_lock:
            return self._connected

    @property
    def connection_revision(self) -> int:
        with self._connection_lock:
            return self._connection_revision

    def mark_connected_if_unchanged(self, expected_revision: int) -> bool:
        """Use a successful connect return only if no callback raced it."""
        with self._connection_lock:
            if self._connection_revision != int(expected_revision):
                return self._connected
            self._connected = True
            self._connection_revision += 1
            self.last_disconnect_reason = None
            return True

    def bind_gate_controller(self, controller: Any) -> None:
        """Bind an object exposing ``set_gate(bool)`` for fail-closed stops."""
        if not callable(getattr(controller, "set_gate", None)):
            raise TypeError("gate controller must implement set_gate(bool)")
        if controller not in self._gate_controllers:
            self._gate_controllers.append(controller)

    def on_connected(self) -> None:
        with self._connection_lock:
            self._connected = True
            self._connection_revision += 1
            self.last_disconnect_reason = None

    def on_disconnected(self) -> None:
        with self._connection_lock:
            self._connected = False
            self._connection_revision += 1
            self.last_disconnect_reason = "qmt_disconnected"
        for controller in tuple(self._gate_controllers):
            try:
                controller.set_gate(False, reason="qmt_disconnected")
            except TypeError:
                controller.set_gate(False)
        logger.error("QMT disconnected; managed orders require reconciliation before new submissions")

    def on_account_status(self, status: Any) -> None:
        logger.info("QMT account status: %s", getattr(status, "status", status))

    def on_stock_order(self, order: Any) -> None:
        try:
            self.manager.apply_order_update(
                BrokerOrderSnapshot(
                    broker_order_id=str(getattr(order, "order_id", "")) or None,
                    broker_order_sys_id=str(getattr(order, "order_sysid", "")) or None,
                    client_order_id=getattr(order, "order_remark", "") or None,
                    symbol=str(getattr(order, "stock_code", "")),
                    status=_order_status(getattr(order, "order_status", 255)),
                    filled_quantity=int(getattr(order, "traded_volume", 0) or 0),
                    avg_fill_price=float(getattr(order, "traded_price", 0) or 0) or None,
                    reject_reason=getattr(order, "status_msg", "") or None,
                )
            )
        except Exception:
            logger.exception("failed to consume QMT order callback")

    def on_stock_trade(self, trade: Any) -> None:
        try:
            self.manager.apply_fill(
                FillEvent(
                    fill_id=str(getattr(trade, "traded_id", "")),
                    broker_order_id=str(getattr(trade, "order_id", "")) or None,
                    broker_order_sys_id=str(getattr(trade, "order_sysid", "")) or None,
                    client_order_id=getattr(trade, "order_remark", "") or None,
                    symbol=str(getattr(trade, "stock_code", "")),
                    quantity=int(getattr(trade, "traded_volume", 0) or 0),
                    price=float(getattr(trade, "traded_price", 0) or 0),
                    execution_timestamp=str(getattr(trade, "traded_time", "")),
                )
            )
        except Exception:
            logger.exception("failed to consume QMT trade callback")

    def on_order_error(self, error: Any) -> None:
        try:
            record = self.manager._resolve(
                getattr(error, "order_remark", None),
                str(getattr(error, "order_id", "")) or None,
                None,
            )
            if record is not None:
                self.manager.apply_order_update(
                    BrokerOrderSnapshot(
                        broker_order_id=str(getattr(error, "order_id", "")) or None,
                        client_order_id=getattr(error, "order_remark", "") or None,
                        symbol=record.symbol,
                        status=OrderStatus.REJECTED,
                        reject_reason=getattr(error, "error_msg", "") or "order_error",
                    )
                )
        except Exception:
            logger.exception("failed to consume QMT order error callback")

    def on_cancel_error(self, error: Any) -> None:
        logger.error(
            "QMT cancel error order_id=%s message=%s",
            getattr(error, "order_id", None),
            getattr(error, "error_msg", None),
        )

    def on_stock_asset(self, asset: Any) -> None:
        pass

    def on_stock_position(self, position: Any) -> None:
        pass

    def on_order_stock_async_response(self, response: Any) -> None:
        pass

    def on_smt_appointment_async_response(self, response: Any) -> None:
        pass

    def on_cancel_order_stock_async_response(self, response: Any) -> None:
        pass
