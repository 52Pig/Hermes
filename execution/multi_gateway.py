"""Optional execution channels behind the Hermes broker contract.

Vendor SDKs are intentionally injected instead of imported here.  PTrade,
FIX and CTP deployments differ by broker and version; this module standardises
the Hermes boundary without pretending that a broker login or entitlement is
available.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Any, Dict, Iterable, Mapping, Optional

from .order_protocol import (
    BrokerOrderSnapshot,
    FillEvent,
    OrderRecord,
    OrderSide,
    OrderStatus,
    SubmissionResult,
)
from .account_protocol import AccountSnapshot, PositionSnapshot


def _value(result: Any, *names: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        for name in names:
            if name in result:
                return result[name]
    for name in names:
        if hasattr(result, name):
            return getattr(result, name)
    return default


def _submission(result: Any) -> SubmissionResult:
    if isinstance(result, SubmissionResult):
        return result
    broker_id = _value(result, "broker_order_id", "order_id", "order_ref", "order_sys_id")
    if broker_id is None and isinstance(result, (int, str)):
        broker_id = result if str(result) not in {"", "-1", "0"} else None
    accepted = bool(_value(result, "accepted", "success", default=broker_id is not None))
    return SubmissionResult(
        accepted=accepted,
        broker_order_id=str(broker_id) if broker_id is not None else None,
        broker_order_sys_id=(str(_value(result, "broker_order_sys_id", "order_sys_id"))
                             if _value(result, "broker_order_sys_id", "order_sys_id") is not None else None),
        reason=_value(result, "reason", "error_msg", "message"),
    )


class GatewayRouter:
    """Route intents to a channel selected by metadata or account defaults."""

    def __init__(self, gateways: Mapping[str, Any], *, default_channel: str = "",
                 account_channels: Optional[Mapping[str, str]] = None) -> None:
        self.gateways = {str(key).lower(): value for key, value in gateways.items()}
        self.default_channel = str(default_channel).lower()
        self.account_channels = {
            str(account): str(channel).lower()
            for account, channel in (account_channels or {}).items()
        }
        self.last_query_errors: Dict[str, str] = {}

    def _account_gateway(self, account_id: str) -> Any:
        channel = self.account_channels.get(str(account_id), self.default_channel)
        gateway = self.gateways.get(channel)
        if gateway is None:
            raise KeyError(f"no account gateway configured for {account_id!r} on {channel!r}")
        return gateway

    def query_account(self, account_id: str) -> AccountSnapshot:
        query = getattr(self._account_gateway(account_id), "query_account", None)
        if query is None:
            raise RuntimeError("configured gateway does not implement query_account")
        return query(account_id)

    def query_positions(self, account_id: str) -> Iterable[PositionSnapshot]:
        query = getattr(self._account_gateway(account_id), "query_positions", None)
        if query is None:
            raise RuntimeError("configured gateway does not implement query_positions")
        return query(account_id)

    def channel_for(self, intent) -> str:
        channel = str(intent.metadata.get("execution_channel") or self.default_channel).lower()
        if not channel or channel not in self.gateways:
            raise KeyError(f"no execution gateway configured for channel {channel!r}")
        return str(channel)

    def register(self, channel: str, gateway: Any) -> None:
        channel = str(channel).lower()
        if not channel or channel in self.gateways:
            raise ValueError(f"gateway channel already registered or empty: {channel!r}")
        if not callable(getattr(gateway, "submit", None)) or not callable(getattr(gateway, "cancel", None)):
            raise TypeError("gateway must implement submit() and cancel()")
        self.gateways[channel] = gateway

    def bind_manager(self, manager: Any) -> None:
        for gateway in self.gateways.values():
            bind = getattr(gateway, "bind_manager", None)
            if bind:
                bind(manager)

    def prepare_intent(self, intent):
        channel = self.channel_for(intent)
        metadata = dict(intent.metadata)
        metadata["execution_channel"] = channel
        return replace(intent, metadata=metadata)

    def submit(self, intent) -> SubmissionResult:
        return self.gateways[self.channel_for(intent)].submit(intent)

    def cancel(self, record: OrderRecord) -> bool:
        channel = str(record.metadata.get("execution_channel") or self.default_channel).lower()
        gateway = self.gateways.get(str(channel))
        return bool(gateway and gateway.cancel(record))

    def clear_query_errors(self) -> None:
        self.last_query_errors.clear()

    def query_orders(self, *, cancelable_only: bool = False) -> Iterable[BrokerOrderSnapshot]:
        snapshots = []
        for channel, gateway in self.gateways.items():
            query = getattr(gateway, "query_orders", None)
            if query:
                try:
                    snapshots.extend(query(cancelable_only=cancelable_only) or [])
                except Exception as exc:
                    # Keep other channels available for reconciliation.  The
                    # caller must inspect last_query_errors before opening the
                    # live order gate.
                    self.last_query_errors[channel] = f"{type(exc).__name__}: {exc}"
        return snapshots

    def query_fills(self) -> Iterable[FillEvent]:
        fills = []
        for channel, gateway in self.gateways.items():
            query = getattr(gateway, "query_fills", None)
            if query:
                try:
                    fills.extend(query() or [])
                except Exception as exc:
                    self.last_query_errors[channel] = f"{type(exc).__name__}: {exc}"
        return fills


class PaperGateway:
    """Deterministic paper channel for integration and recovery drills."""

    def __init__(self, *, fill_immediately: bool = False) -> None:
        self.fill_immediately = fill_immediately
        self._counter = itertools.count(1)
        self.orders: Dict[str, Any] = {}
        self.statuses: Dict[str, OrderStatus] = {}
        self._manager = None

    def bind_manager(self, manager: Any) -> None:
        self._manager = manager

    def submit(self, intent) -> SubmissionResult:
        broker_id = f"PAPER-{next(self._counter)}"
        self.orders[broker_id] = intent
        self.statuses[broker_id] = OrderStatus.PENDING
        if self._manager is not None:
            self._manager.apply_order_update(BrokerOrderSnapshot(
                broker_order_id=broker_id, client_order_id=intent.client_order_id,
                symbol=intent.symbol, status=OrderStatus.PENDING,
            ))
        return SubmissionResult(accepted=True, broker_order_id=broker_id)

    def cancel(self, record: OrderRecord) -> bool:
        if record.broker_order_id not in self.orders:
            return False
        self.statuses[record.broker_order_id] = OrderStatus.CANCELLED
        if self._manager is not None:
            intent = self.orders[record.broker_order_id]
            self._manager.apply_order_update(BrokerOrderSnapshot(
                broker_order_id=record.broker_order_id,
                client_order_id=intent.client_order_id,
                symbol=intent.symbol, status=OrderStatus.CANCELLED,
            ))
        return True

    def fill(self, broker_order_id: str, quantity: int, price: float,
             *, fill_id: Optional[str] = None, execution_timestamp: str = "") -> OrderStatus:
        if self._manager is None:
            raise RuntimeError("paper gateway is not bound to an OrderManager")
        intent = self.orders[broker_order_id]
        record = self._manager.store.find_by_broker_id(broker_order_id)
        if record is None:
            raise KeyError(broker_order_id)
        applied = self._manager.apply_fill(FillEvent(
            fill_id=fill_id or f"{broker_order_id}:{record.filled_quantity + int(quantity)}",
            client_order_id=intent.client_order_id,
            broker_order_id=broker_order_id,
            symbol=intent.symbol,
            quantity=quantity,
            price=price,
            execution_timestamp=execution_timestamp,
        ))
        self.statuses[broker_order_id] = applied.status
        return applied.status

    def query_orders(self, *, cancelable_only: bool = False) -> Iterable[BrokerOrderSnapshot]:
        for broker_id, intent in self.orders.items():
            status = self.statuses[broker_id]
            if cancelable_only and status not in {
                OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED,
                OrderStatus.CANCEL_REQUESTED,
            }:
                continue
            yield BrokerOrderSnapshot(
                broker_order_id=broker_id,
                client_order_id=intent.client_order_id,
                symbol=intent.symbol,
                status=status,
            )


class ManualApprovalGateway:
    """Deliberately fail-closed channel for human-confirmed execution."""

    def submit(self, intent) -> SubmissionResult:
        return SubmissionResult(accepted=False, reason="manual_approval_required")

    def cancel(self, record: OrderRecord) -> bool:
        return False


class _ClientGateway:
    def __init__(self, client: Any, *, channel: str) -> None:
        self.client = client
        self.channel = channel
        self._manager = None

    def bind_manager(self, manager: Any) -> None:
        self._manager = manager
        register = getattr(self.client, "register_callbacks", None)
        if register is None:
            return
        register(
            on_order=self._on_order_event,
            on_fill=self._on_fill_event,
        )

    def _on_order_event(self, event: Any) -> None:
        if self._manager is None:
            return
        if not isinstance(event, BrokerOrderSnapshot):
            raise TypeError("gateway client order callback must emit BrokerOrderSnapshot")
        self._manager.apply_order_update(event)

    def _on_fill_event(self, event: Any) -> None:
        if self._manager is None:
            return
        if not isinstance(event, FillEvent):
            raise TypeError("gateway client fill callback must emit FillEvent")
        self._manager.apply_fill(event)

    def submit(self, intent) -> SubmissionResult:
        method = getattr(self.client, "submit_order", None) or getattr(self.client, "order_insert", None)
        if method is None:
            raise RuntimeError(f"{self.channel} client must provide submit_order or order_insert")
        result = method(self.request(intent))
        return _submission(result)

    def cancel(self, record: OrderRecord) -> bool:
        method = getattr(self.client, "cancel_order", None) or getattr(self.client, "order_cancel", None)
        if method is None or not record.broker_order_id:
            return False
        result = method(record.broker_order_id)
        return bool(_value(result, "accepted", "success", default=result in (0, True)))

    def query_orders(self, *, cancelable_only: bool = False) -> Iterable[BrokerOrderSnapshot]:
        method = getattr(self.client, "query_orders", None) or getattr(self.client, "get_orders", None)
        if method is None:
            raise RuntimeError(f"{self.channel} client must provide query_orders or get_orders")
        for order in method(cancelable_only=cancelable_only) or []:
            raw_status = str(_value(order, "status", "order_status", "OrderStatus", default="UNKNOWN")).upper()
            aliases = {
                "NEW": OrderStatus.PENDING,
                "SUBMITTED": OrderStatus.PENDING,
                "ACCEPTED": OrderStatus.PENDING,
                "PARTIAL": OrderStatus.PARTIALLY_FILLED,
                "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
                "FILLED": OrderStatus.FILLED,
                "CANCEL_PENDING": OrderStatus.CANCEL_REQUESTED,
                "CANCEL_REQUESTED": OrderStatus.CANCEL_REQUESTED,
                "CANCELED": OrderStatus.CANCELLED,
                "CANCELLED": OrderStatus.CANCELLED,
                "REJECTED": OrderStatus.REJECTED,
                # Common CTP OrderStatus values and vn.py names.
                "0": OrderStatus.PENDING,
                "1": OrderStatus.PARTIALLY_FILLED,
                "2": OrderStatus.FILLED,
                "5": OrderStatus.CANCELLED,
                "ALLTRADED": OrderStatus.FILLED,
                "PARTTRADED": OrderStatus.PARTIALLY_FILLED,
                "NOTRADENOTQUEUEING": OrderStatus.CANCELLED,
                "NOTRADEQUEUEING": OrderStatus.PENDING,
            }
            try:
                status = aliases[raw_status]
            except KeyError:
                status = OrderStatus.UNKNOWN
            broker_id = _value(order, "broker_order_id", "order_id", "vt_orderid", "order_ref", "OrderRef")
            client_id = _value(order, "client_order_id", "client_id", "reference", "order_remark", "OrderRef")
            yield BrokerOrderSnapshot(
                broker_order_id=str(broker_id) if broker_id is not None else None,
                broker_order_sys_id=(str(_value(order, "broker_order_sys_id", "order_sys_id", "OrderSysID"))
                                     if _value(order, "broker_order_sys_id", "order_sys_id", "OrderSysID") is not None else None),
                client_order_id=str(client_id) if client_id else None,
                symbol=str(_value(order, "symbol", "instrument", "vt_symbol", "InstrumentID", default="")),
                status=status,
                filled_quantity=int(_value(order, "filled_quantity", "traded_volume", "volume_traded", "VolumeTraded", default=0) or 0),
                avg_fill_price=float(_value(order, "avg_fill_price", "traded_price", "avg_price", "Price", default=0) or 0) or None,
                reject_reason=_value(order, "reject_reason", "status_msg", "error_msg"),
            )

    def query_fills(self) -> Iterable[FillEvent]:
        method = getattr(self.client, "query_fills", None) or getattr(self.client, "get_fills", None)
        if method is None:
            raise RuntimeError(f"{self.channel} client must provide query_fills or get_fills")
        for fill in method() or []:
            fill_id = _value(fill, "fill_id", "trade_id", "traded_id", "exec_id", "TradeID")
            broker_id = _value(fill, "broker_order_id", "order_id", "vt_orderid", "order_ref", "OrderRef")
            if not fill_id or broker_id is None:
                continue
            quantity = int(_value(fill, "quantity", "volume", "traded_volume", "Volume", default=0) or 0)
            price = float(_value(fill, "price", "traded_price", "fill_price", "Price", default=0) or 0)
            if quantity <= 0 or price <= 0:
                continue
            yield FillEvent(
                fill_id=str(fill_id), broker_order_id=str(broker_id),
                client_order_id=_value(fill, "client_order_id", "client_id", "reference", "order_remark"),
                symbol=str(_value(fill, "symbol", "instrument", "vt_symbol", "InstrumentID", default="")),
                quantity=quantity, price=price,
                execution_timestamp=str(_value(fill, "execution_timestamp", "trade_time", "traded_time", "TradeTime", default="")),
                fee=float(_value(fill, "fee", "commission", "Commission", default=0) or 0),
            )

    def query_account(self, account_id: str) -> AccountSnapshot:
        method = getattr(self.client, "query_account", None) or getattr(self.client, "get_account", None)
        if method is None:
            raise RuntimeError(f"{self.channel} client must provide query_account or get_account")
        raw = method(account_id)
        cash = float(_value(raw, "cash", "balance", "available_cash", default=0) or 0)
        available = float(_value(raw, "available_cash", "available", "cash", default=0) or 0)
        equity = float(_value(raw, "equity", "total_asset", "net_asset", default=0) or 0)
        return AccountSnapshot(
            account_id=account_id,
            currency=str(_value(raw, "currency", "currency_id", default="CNY")),
            cash=cash,
            available_cash=available,
            equity=equity,
            margin_used=float(_value(raw, "margin_used", "curr_margin", "margin", default=0) or 0),
            frozen_cash=float(_value(raw, "frozen_cash", "commission_frozen", default=0) or 0),
            metadata={"channel": self.channel},
        )

    def query_positions(self, account_id: str) -> Iterable[PositionSnapshot]:
        method = getattr(self.client, "query_positions", None) or getattr(self.client, "get_positions", None)
        if method is None:
            raise RuntimeError(f"{self.channel} client must provide query_positions or get_positions")
        for position in method(account_id) or []:
            quantity = float(_value(position, "quantity", "volume", "position", default=0) or 0)
            yield PositionSnapshot(
                account_id=account_id,
                symbol=str(_value(position, "symbol", "instrument", "vt_symbol", default="")),
                quantity=quantity,
                available_quantity=float(_value(position, "available_quantity", "can_use_volume", "volume", default=quantity) or 0),
                average_price=float(_value(position, "average_price", "open_price", "avg_price", default=0) or 0),
                market_value=float(_value(position, "market_value", "position_value", "position_cost", default=0) or 0),
                side=str(_value(position, "side", "direction", default="LONG")).upper(),
                today_quantity=float(_value(position, "today_quantity", "today_volume", default=0) or 0),
                yesterday_quantity=float(_value(position, "yesterday_quantity", "yd_volume", default=0) or 0),
                metadata={"channel": self.channel},
            )

    def request(self, intent) -> Dict[str, Any]:
        return {
            "client_order_id": intent.client_order_id,
            "account_id": intent.account_id,
            "symbol": intent.symbol,
            "side": intent.side,
            "quantity": intent.quantity,
            "order_type": intent.order_type,
            "limit_price": intent.limit_price,
            **dict(intent.metadata),
        }


class PTradeGateway(_ClientGateway):
    def __init__(self, client: Any) -> None:
        super().__init__(client, channel="ptrade")


class FixGateway(_ClientGateway):
    def __init__(self, client: Any) -> None:
        super().__init__(client, channel="fix")


class BrokerApiGateway(_ClientGateway):
    def __init__(self, client: Any) -> None:
        super().__init__(client, channel="broker_api")


class BrokerAlgoGateway(_ClientGateway):
    """Broker-native conditional/algo execution API contract."""

    def __init__(self, client: Any) -> None:
        super().__init__(client, channel="broker_algo")

    def submit(self, intent) -> SubmissionResult:
        method = getattr(self.client, "submit_algo", None)
        if method is None:
            return super().submit(intent)
        return _submission(method(self.request(intent)))

    def cancel(self, record: OrderRecord) -> bool:
        method = getattr(self.client, "cancel_algo", None)
        if method is None:
            return super().cancel(record)
        if not record.broker_order_id:
            return False
        result = method(record.broker_order_id)
        return bool(_value(result, "accepted", "success", default=result in (0, True)))


class HostedStrategyGateway(_ClientGateway):
    """Gateway boundary for a broker-hosted strategy execution service."""

    def __init__(self, client: Any) -> None:
        super().__init__(client, channel="hosted_strategy")

    def request(self, intent) -> Dict[str, Any]:
        request = super().request(intent)
        request["task_type"] = "submit_order_intent"
        return request


class CtpGateway(_ClientGateway):
    """CTP adapter contract; vendor-specific CTP client is injected."""

    def __init__(self, client: Any) -> None:
        super().__init__(client, channel="ctp")

    def request(self, intent) -> Dict[str, Any]:
        metadata = dict(intent.metadata)
        for field in ("exchange", "position_effect", "hedge_flag"):
            if not metadata.get(field):
                raise ValueError(f"CTP intent requires metadata.{field}")
        request = super().request(intent)
        side_map = {"BUY": "Buy", "SELL": "Sell"}
        offset_map = {
            "OPEN": "Open", "CLOSE": "Close", "CLOSETODAY": "CloseToday",
            "CLOSE_TODAY": "CloseToday", "CLOSEYESTERDAY": "CloseYesterday",
            "CLOSE_YESTERDAY": "CloseYesterday",
        }
        hedge_map = {"SPECULATION": "Speculation", "ARBITRAGE": "Arbitrage", "HEDGE": "Hedge"}
        price_type = metadata.get("order_price_type")
        if not price_type:
            price_type = "AnyPrice" if intent.order_type == "MARKET" else "LimitPrice"
        position_effect = str(metadata["position_effect"]).upper()
        hedge_flag = str(metadata["hedge_flag"]).upper()
        request.update({
            "exchange": metadata["exchange"],
            "position_effect": metadata["position_effect"],
            "hedge_flag": metadata["hedge_flag"],
            "time_condition": metadata.get("time_condition", "GFD"),
            "volume_condition": metadata.get("volume_condition", "AV"),
            # Names match CTP CThostFtdcInputOrderField.  A broker client can
            # pass this dictionary directly or apply its own enum constants.
            "Direction": metadata.get("Direction", side_map[intent.side]),
            "CombOffsetFlag": metadata.get("CombOffsetFlag", offset_map.get(position_effect, metadata["position_effect"])),
            "CombHedgeFlag": metadata.get("CombHedgeFlag", hedge_map.get(hedge_flag, metadata["hedge_flag"])),
            "OrderPriceType": metadata.get("OrderPriceType", price_type),
            "ExchangeID": metadata.get("ExchangeID", metadata["exchange"]),
            "InstrumentID": metadata.get("InstrumentID", intent.symbol.split(".")[0]),
            "LimitPrice": metadata.get("LimitPrice", intent.limit_price or 0.0),
            "VolumeTotalOriginal": metadata.get("VolumeTotalOriginal", intent.quantity),
            "BrokerID": metadata.get("BrokerID"),
            "InvestorID": metadata.get("InvestorID", intent.account_id),
            "OrderRef": metadata.get("OrderRef", intent.client_order_id),
        })
        return request

    def cancel(self, record: OrderRecord) -> bool:
        if not record.broker_order_id:
            return False
        metadata = dict(record.metadata)
        cancel_method = (
            getattr(self.client, "cancel_order_ctp", None)
            or getattr(self.client, "order_action", None)
        )
        if cancel_method is None:
            return super().cancel(record)
        request = {
            "BrokerID": metadata.get("BrokerID"),
            "InvestorID": metadata.get("InvestorID", record.account_id),
            "InstrumentID": metadata.get("InstrumentID", record.symbol.split(".")[0]),
            "ExchangeID": metadata.get("ExchangeID", metadata.get("exchange")),
            "OrderSysID": record.broker_order_sys_id or record.broker_order_id,
            "OrderRef": metadata.get("OrderRef", record.client_order_id),
            "FrontID": metadata.get("FrontID", 0),
            "SessionID": metadata.get("SessionID", 0),
            "ActionFlag": metadata.get("ActionFlag", "Delete"),
        }
        result = cancel_method(request)
        return bool(_value(result, "accepted", "success", default=result in (0, True)))
