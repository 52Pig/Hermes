"""Durable, idempotent order protocol for Hermes.

The strategy layer creates an :class:`OrderIntent`.  The intent receives a
deterministic ``client_order_id`` before it reaches QMT.  ``OrderManager``
persists that intent in SQLite and applies broker order/fill events exactly
once.  The module deliberately has no dependency on ``xtquant`` so that the
state machine can be tested offline and reused by a simulator.

The broker call can still be ambiguous (for example, the process can stop
after QMT accepted an order but before the response was persisted).  Such an
order is marked ``UNKNOWN`` and is never submitted again automatically.  A
reconciliation pass must first match the deterministic client id or the QMT
order id.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple


def utc_now() -> str:
    """Return a sortable UTC timestamp with an explicit timezone."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    """Local status values; broker-specific numeric values stay in adapters."""

    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


OPEN_STATUSES = {
    OrderStatus.PENDING,
    OrderStatus.PARTIALLY_FILLED,
    OrderStatus.CANCEL_REQUESTED,
    OrderStatus.UNKNOWN,
}
TERMINAL_STATUSES = {
    OrderStatus.FILLED,
    OrderStatus.CANCELLED,
    OrderStatus.REJECTED,
}

# An order update may arrive more than once and may arrive out of order.  The
# graph is intentionally conservative: a terminal order can never become open
# again, and UNKNOWN cannot be silently retried.
_ALLOWED_TRANSITIONS = {
    OrderStatus.PENDING: {
        OrderStatus.PENDING,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCEL_REQUESTED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
        OrderStatus.UNKNOWN,
    },
    OrderStatus.PARTIALLY_FILLED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCEL_REQUESTED,
        OrderStatus.CANCELLED,
        OrderStatus.UNKNOWN,
    },
    OrderStatus.CANCEL_REQUESTED: {
        OrderStatus.CANCEL_REQUESTED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.UNKNOWN,
    },
    OrderStatus.UNKNOWN: {
        OrderStatus.UNKNOWN,
        OrderStatus.PENDING,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
    },
    OrderStatus.FILLED: {OrderStatus.FILLED},
    OrderStatus.CANCELLED: {OrderStatus.CANCELLED},
    OrderStatus.REJECTED: {OrderStatus.REJECTED},
}


class OrderProtocolError(RuntimeError):
    """Base class for protocol failures."""


class InvalidOrderState(OrderProtocolError):
    """Raised when an event attempts an impossible status transition."""


class OrderConflictError(OrderProtocolError):
    """Raised when a deterministic id is reused for a different order."""


class AmbiguousSubmissionError(OrderProtocolError):
    """Raised when a broker call may have been accepted but is not confirmed."""


class ExecutionGateClosed(OrderProtocolError):
    """Raised when the account is not cleared for new broker submissions."""


def _enum_value(value: Any, enum_type: Any) -> str:
    if isinstance(value, enum_type):
        return value.value
    return enum_type(str(value).upper()).value


def _normalise_optional_timestamp(value: Optional[Any]) -> Optional[str]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat(timespec="milliseconds")
    return str(value)


def _stable_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _safe_fragment(value: str, length: int = 18) -> str:
    fragment = re.sub(r"[^A-Za-z0-9]+", "", str(value))
    return (fragment or "NA")[:length]


def make_client_order_id(
    *,
    account_id: str,
    strategy_name: str,
    signal_id: str,
    symbol: str,
    side: OrderSide | str,
    quantity: int,
    order_type: OrderType | str,
    limit_price: Optional[float],
    signal_timestamp: Optional[str] = None,
    execution_channel: Optional[str] = None,
) -> str:
    """Build a deterministic id for one logical order intent.

    The hash identifies the logical signal, rather than a transient quote.
    Re-running a strategy with the same signal produces the same id even when
    a quote changes.  The durable payload then rejects a changed quantity or
    price as a conflict instead of risking a second broker submission.
    """

    payload = {
        "account_id": str(account_id),
        "execution_channel": str(execution_channel or "").lower(),
        "strategy_name": str(strategy_name),
        "signal_id": str(signal_id or ""),
        "symbol": str(symbol).upper(),
        "side": _enum_value(side, OrderSide),
        "order_type": _enum_value(order_type, OrderType),
        "signal_timestamp": _normalise_optional_timestamp(signal_timestamp),
    }
    digest = sha256(_stable_json(payload).encode("utf-8")).hexdigest()[:24]
    return "CO-{}-{}".format(_safe_fragment(strategy_name), digest)


@dataclass(frozen=True)
class OrderIntent:
    """Immutable order request produced by a strategy."""

    account_id: str
    strategy_name: str
    symbol: str
    side: OrderSide | str
    quantity: int
    order_type: OrderType | str = OrderType.LIMIT
    limit_price: Optional[float] = None
    signal_id: str = ""
    prediction_date: Optional[str] = None
    feature_cutoff_timestamp: Optional[str] = None
    prediction_timestamp: Optional[str] = None
    signal_timestamp: Optional[str] = None
    earliest_fill_timestamp: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", _enum_value(self.side, OrderSide))
        object.__setattr__(self, "order_type", _enum_value(self.order_type, OrderType))
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        object.__setattr__(self, "account_id", str(self.account_id).strip())
        object.__setattr__(self, "strategy_name", str(self.strategy_name).strip())
        if not self.account_id:
            raise ValueError("account_id is required")
        if not self.strategy_name:
            raise ValueError("strategy_name is required")
        if not self.symbol:
            raise ValueError("symbol is required")
        if int(self.quantity) <= 0:
            raise ValueError("quantity must be positive")
        object.__setattr__(self, "quantity", int(self.quantity))
        if self.order_type == OrderType.LIMIT:
            if self.limit_price is None or float(self.limit_price) <= 0:
                raise ValueError("a positive limit_price is required for LIMIT orders")
            object.__setattr__(self, "limit_price", float(self.limit_price))
        elif self.limit_price is not None and float(self.limit_price) < 0:
            raise ValueError("limit_price cannot be negative")
        for name in (
            "prediction_date",
            "feature_cutoff_timestamp",
            "prediction_timestamp",
            "signal_timestamp",
            "earliest_fill_timestamp",
        ):
            object.__setattr__(self, name, _normalise_optional_timestamp(getattr(self, name)))

    @property
    def client_order_id(self) -> str:
        return make_client_order_id(
            account_id=self.account_id,
            strategy_name=self.strategy_name,
            signal_id=self.signal_id,
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            order_type=self.order_type,
            limit_price=self.limit_price,
            signal_timestamp=self.signal_timestamp,
            execution_channel=self.metadata.get("execution_channel"),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["side"] = self.side.value if isinstance(self.side, Enum) else self.side
        result["order_type"] = self.order_type.value if isinstance(self.order_type, Enum) else self.order_type
        result["metadata"] = dict(self.metadata)
        result["client_order_id"] = self.client_order_id
        return result


@dataclass
class OrderRecord:
    """Durable local order record."""

    client_order_id: str
    account_id: str
    strategy_name: str
    symbol: str
    side: str
    quantity: int
    order_type: str
    limit_price: Optional[float]
    signal_id: str = ""
    prediction_date: Optional[str] = None
    feature_cutoff_timestamp: Optional[str] = None
    prediction_timestamp: Optional[str] = None
    signal_timestamp: Optional[str] = None
    earliest_fill_timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: OrderStatus = OrderStatus.PENDING
    broker_order_id: Optional[str] = None
    broker_order_sys_id: Optional[str] = None
    filled_quantity: int = 0
    avg_fill_price: Optional[float] = None
    reject_reason: Optional[str] = None
    cancel_reason: Optional[str] = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def from_intent(cls, intent: OrderIntent, now: Optional[str] = None) -> "OrderRecord":
        timestamp = now or utc_now()
        return cls(
            client_order_id=intent.client_order_id,
            account_id=intent.account_id,
            strategy_name=intent.strategy_name,
            symbol=intent.symbol,
            side=str(intent.side),
            quantity=intent.quantity,
            order_type=str(intent.order_type),
            limit_price=intent.limit_price,
            signal_id=intent.signal_id,
            prediction_date=intent.prediction_date,
            feature_cutoff_timestamp=intent.feature_cutoff_timestamp,
            prediction_timestamp=intent.prediction_timestamp,
            signal_timestamp=intent.signal_timestamp,
            earliest_fill_timestamp=intent.earliest_fill_timestamp,
            metadata=dict(intent.metadata),
            created_at=timestamp,
            updated_at=timestamp,
        )

    def intent(self) -> OrderIntent:
        return OrderIntent(
            account_id=self.account_id,
            strategy_name=self.strategy_name,
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            order_type=self.order_type,
            limit_price=self.limit_price,
            signal_id=self.signal_id,
            prediction_date=self.prediction_date,
            feature_cutoff_timestamp=self.feature_cutoff_timestamp,
            prediction_timestamp=self.prediction_timestamp,
            signal_timestamp=self.signal_timestamp,
            earliest_fill_timestamp=self.earliest_fill_timestamp,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class SubmissionResult:
    accepted: bool
    broker_order_id: Optional[str] = None
    broker_order_sys_id: Optional[str] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class FillEvent:
    fill_id: str
    symbol: str
    quantity: int
    price: float
    execution_timestamp: str
    client_order_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    broker_order_sys_id: Optional[str] = None
    fee: float = 0.0

    def __post_init__(self) -> None:
        if not self.fill_id:
            raise ValueError("fill_id is required")
        if int(self.quantity) <= 0:
            raise ValueError("fill quantity must be positive")
        if float(self.price) <= 0:
            raise ValueError("fill price must be positive")
        object.__setattr__(self, "quantity", int(self.quantity))
        object.__setattr__(self, "price", float(self.price))


@dataclass(frozen=True)
class BrokerOrderSnapshot:
    broker_order_id: Optional[str]
    symbol: str
    status: OrderStatus | str
    filled_quantity: int = 0
    avg_fill_price: Optional[float] = None
    client_order_id: Optional[str] = None
    broker_order_sys_id: Optional[str] = None
    reject_reason: Optional[str] = None
    cancel_reason: Optional[str] = None


class BrokerGateway(Protocol):
    def submit(self, intent: OrderIntent) -> SubmissionResult:
        ...

    def cancel(self, record: OrderRecord) -> bool:
        ...

    def query_orders(self, *, cancelable_only: bool = False) -> Iterable[BrokerOrderSnapshot]:
        ...

    def query_fills(self) -> Iterable[FillEvent]:
        ...


def _normalise_status(value: Any) -> OrderStatus:
    if isinstance(value, OrderStatus):
        return value
    if isinstance(value, int):
        # QMT values are intentionally mirrored here as a small compatibility
        # map; the adapter remains responsible for translating other brokers.
        return {
            48: OrderStatus.PENDING,
            49: OrderStatus.PENDING,
            50: OrderStatus.PENDING,
            51: OrderStatus.CANCEL_REQUESTED,
            52: OrderStatus.CANCEL_REQUESTED,
            53: OrderStatus.CANCELLED,
            54: OrderStatus.CANCELLED,
            55: OrderStatus.PARTIALLY_FILLED,
            56: OrderStatus.FILLED,
            57: OrderStatus.REJECTED,
            255: OrderStatus.UNKNOWN,
        }.get(value, OrderStatus.UNKNOWN)
    value = str(value).upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "SUBMITTED": OrderStatus.PENDING,
        "ACCEPTED": OrderStatus.PENDING,
        "NEW": OrderStatus.PENDING,
        "OPEN": OrderStatus.PENDING,
        "PARTIAL": OrderStatus.PARTIALLY_FILLED,
        "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
        "PART_SUCC": OrderStatus.PARTIALLY_FILLED,
        "SUCCESS": OrderStatus.FILLED,
        "SUCCEEDED": OrderStatus.FILLED,
        "FILLED": OrderStatus.FILLED,
        "CANCELED": OrderStatus.CANCELLED,
        "CANCELLED": OrderStatus.CANCELLED,
        "REJECT": OrderStatus.REJECTED,
        "REJECTED": OrderStatus.REJECTED,
        "JUNK": OrderStatus.REJECTED,
        "UNKNOWN": OrderStatus.UNKNOWN,
    }
    return aliases.get(value, OrderStatus.UNKNOWN)


class SQLiteOrderStore:
    """Small transactional store suitable for one Hermes account process."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.RLock()
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        # A single account should normally be owned by one Hermes process, but
        # restart/reconciliation tools can overlap briefly.  WAL plus a busy
        # timeout lets the second writer wait for the first transaction rather
        # than failing immediately with ``database is locked``.
        self._conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    client_order_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    order_type TEXT NOT NULL,
                    limit_price REAL,
                    signal_id TEXT NOT NULL,
                    prediction_date TEXT,
                    feature_cutoff_timestamp TEXT,
                    prediction_timestamp TEXT,
                    signal_timestamp TEXT,
                    earliest_fill_timestamp TEXT,
                    metadata_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    broker_order_id TEXT,
                    broker_order_sys_id TEXT,
                    filled_quantity INTEGER NOT NULL,
                    avg_fill_price REAL,
                    reject_reason TEXT,
                    cancel_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_orders_broker_id
                    ON orders (broker_order_id);
                CREATE INDEX IF NOT EXISTS idx_orders_broker_sys_id
                    ON orders (broker_order_sys_id);
                CREATE INDEX IF NOT EXISTS idx_orders_status
                    ON orders (status);
                CREATE TABLE IF NOT EXISTS fills (
                    fill_id TEXT PRIMARY KEY,
                    client_order_id TEXT NOT NULL,
                    broker_order_id TEXT,
                    broker_order_sys_id TEXT,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    execution_timestamp TEXT NOT NULL,
                    fee REAL NOT NULL,
                    received_at TEXT NOT NULL,
                    FOREIGN KEY (client_order_id) REFERENCES orders(client_order_id)
                );
                CREATE INDEX IF NOT EXISTS idx_fills_order
                    ON fills (client_order_id);
                CREATE TABLE IF NOT EXISTS order_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_order_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (client_order_id) REFERENCES orders(client_order_id)
                );
                """
            )
            # Schema version is intentionally additive so old order databases
            # remain readable while migrations can be introduced explicitly.
            self._conn.execute("PRAGMA user_version = 1")

    @staticmethod
    def _record_values(record: OrderRecord) -> Tuple[Any, ...]:
        return (
            record.client_order_id,
            record.account_id,
            record.strategy_name,
            record.symbol,
            record.side,
            record.quantity,
            record.order_type,
            record.limit_price,
            record.signal_id,
            record.prediction_date,
            record.feature_cutoff_timestamp,
            record.prediction_timestamp,
            record.signal_timestamp,
            record.earliest_fill_timestamp,
            _stable_json(record.metadata),
            record.status.value,
            record.broker_order_id,
            record.broker_order_sys_id,
            record.filled_quantity,
            record.avg_fill_price,
            record.reject_reason,
            record.cancel_reason,
            record.created_at,
            record.updated_at,
        )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> OrderRecord:
        return OrderRecord(
            client_order_id=row["client_order_id"],
            account_id=row["account_id"],
            strategy_name=row["strategy_name"],
            symbol=row["symbol"],
            side=row["side"],
            quantity=int(row["quantity"]),
            order_type=row["order_type"],
            limit_price=row["limit_price"],
            signal_id=row["signal_id"],
            prediction_date=row["prediction_date"],
            feature_cutoff_timestamp=row["feature_cutoff_timestamp"],
            prediction_timestamp=row["prediction_timestamp"],
            signal_timestamp=row["signal_timestamp"],
            earliest_fill_timestamp=row["earliest_fill_timestamp"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            status=OrderStatus(row["status"]),
            broker_order_id=row["broker_order_id"],
            broker_order_sys_id=row["broker_order_sys_id"],
            filled_quantity=int(row["filled_quantity"]),
            avg_fill_price=row["avg_fill_price"],
            reject_reason=row["reject_reason"],
            cancel_reason=row["cancel_reason"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get(self, client_order_id: str) -> Optional[OrderRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def insert(self, record: OrderRecord) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO orders (
                    client_order_id, account_id, strategy_name, symbol, side,
                    quantity, order_type, limit_price, signal_id, prediction_date,
                    feature_cutoff_timestamp, prediction_timestamp, signal_timestamp,
                    earliest_fill_timestamp, metadata_json, status, broker_order_id,
                    broker_order_sys_id, filled_quantity, avg_fill_price, reject_reason,
                    cancel_reason, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._record_values(record),
            )

    def insert_with_event(self, record: OrderRecord, event_type: str, payload: Mapping[str, Any]) -> None:
        """Insert an order and its audit event in one SQLite transaction."""
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO orders (
                    client_order_id, account_id, strategy_name, symbol, side,
                    quantity, order_type, limit_price, signal_id, prediction_date,
                    feature_cutoff_timestamp, prediction_timestamp, signal_timestamp,
                    earliest_fill_timestamp, metadata_json, status, broker_order_id,
                    broker_order_sys_id, filled_quantity, avg_fill_price, reject_reason,
                    cancel_reason, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._record_values(record),
            )
            self._conn.execute(
                "INSERT INTO order_events (client_order_id, event_type, event_json, created_at) VALUES (?, ?, ?, ?)",
                (record.client_order_id, event_type, _stable_json(dict(payload)), utc_now()),
            )

    def save(self, record: OrderRecord) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE orders SET account_id=?, strategy_name=?, symbol=?, side=?,
                    quantity=?, order_type=?, limit_price=?, signal_id=?, prediction_date=?,
                    feature_cutoff_timestamp=?, prediction_timestamp=?, signal_timestamp=?,
                    earliest_fill_timestamp=?, metadata_json=?, status=?, broker_order_id=?,
                    broker_order_sys_id=?, filled_quantity=?, avg_fill_price=?, reject_reason=?,
                    cancel_reason=?, created_at=?, updated_at=?
                WHERE client_order_id=?
                """,
                self._record_values(record)[1:] + (record.client_order_id,),
            )

    def save_with_event(self, record: OrderRecord, event_type: str, payload: Mapping[str, Any]) -> None:
        """Update an order and append its audit event atomically."""
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE orders SET account_id=?, strategy_name=?, symbol=?, side=?,
                    quantity=?, order_type=?, limit_price=?, signal_id=?, prediction_date=?,
                    feature_cutoff_timestamp=?, prediction_timestamp=?, signal_timestamp=?,
                    earliest_fill_timestamp=?, metadata_json=?, status=?, broker_order_id=?,
                    broker_order_sys_id=?, filled_quantity=?, avg_fill_price=?, reject_reason=?,
                    cancel_reason=?, created_at=?, updated_at=?
                WHERE client_order_id=?
                """,
                self._record_values(record)[1:] + (record.client_order_id,),
            )
            self._conn.execute(
                "INSERT INTO order_events (client_order_id, event_type, event_json, created_at) VALUES (?, ?, ?, ?)",
                (record.client_order_id, event_type, _stable_json(dict(payload)), utc_now()),
            )

    def add_event(self, client_order_id: str, event_type: str, payload: Mapping[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO order_events (client_order_id, event_type, event_json, created_at) VALUES (?, ?, ?, ?)",
                (client_order_id, event_type, _stable_json(dict(payload)), utc_now()),
            )

    def find_by_broker_id(self, broker_order_id: str) -> Optional[OrderRecord]:
        if not broker_order_id:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM orders WHERE broker_order_id = ? ORDER BY updated_at DESC LIMIT 1",
                (str(broker_order_id),),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def find_by_broker_sys_id(self, broker_order_sys_id: str) -> Optional[OrderRecord]:
        if not broker_order_sys_id:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM orders WHERE broker_order_sys_id = ? ORDER BY updated_at DESC LIMIT 1",
                (str(broker_order_sys_id),),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_open(self) -> List[OrderRecord]:
        placeholders = ",".join("?" for _ in OPEN_STATUSES)
        values = tuple(status.value for status in OPEN_STATUSES)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM orders WHERE status IN ({placeholders}) ORDER BY created_at",
                values,
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_all(self) -> List[OrderRecord]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM orders ORDER BY created_at").fetchall()
        return [self._row_to_record(row) for row in rows]

    def add_fill(self, event: FillEvent, client_order_id: str) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO fills (
                    fill_id, client_order_id, broker_order_id, broker_order_sys_id,
                    symbol, quantity, price, execution_timestamp, fee, received_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.fill_id,
                    client_order_id,
                    event.broker_order_id,
                    event.broker_order_sys_id,
                    event.symbol,
                    event.quantity,
                    event.price,
                    event.execution_timestamp,
                    event.fee,
                    utc_now(),
                ),
            )
        return cursor.rowcount == 1

    def has_fill(self, fill_id: str) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM fills WHERE fill_id = ?", (fill_id,)).fetchone()
        return row is not None

    def add_fill_and_save(
        self,
        event: FillEvent,
        client_order_id: str,
        record: OrderRecord,
        payload: Mapping[str, Any],
    ) -> bool:
        """Persist a fill, order aggregate and audit event atomically."""
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO fills (
                    fill_id, client_order_id, broker_order_id, broker_order_sys_id,
                    symbol, quantity, price, execution_timestamp, fee, received_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.fill_id,
                    client_order_id,
                    event.broker_order_id,
                    event.broker_order_sys_id,
                    event.symbol,
                    event.quantity,
                    event.price,
                    event.execution_timestamp,
                    event.fee,
                    utc_now(),
                ),
            )
            if cursor.rowcount != 1:
                return False
            self._conn.execute(
                """
                UPDATE orders SET account_id=?, strategy_name=?, symbol=?, side=?,
                    quantity=?, order_type=?, limit_price=?, signal_id=?, prediction_date=?,
                    feature_cutoff_timestamp=?, prediction_timestamp=?, signal_timestamp=?,
                    earliest_fill_timestamp=?, metadata_json=?, status=?, broker_order_id=?,
                    broker_order_sys_id=?, filled_quantity=?, avg_fill_price=?, reject_reason=?,
                    cancel_reason=?, created_at=?, updated_at=?
                WHERE client_order_id=?
                """,
                self._record_values(record)[1:] + (record.client_order_id,),
            )
            self._conn.execute(
                "INSERT INTO order_events (client_order_id, event_type, event_json, created_at) VALUES (?, ?, ?, ?)",
                (client_order_id, "fill", _stable_json(dict(payload)), utc_now()),
            )
        return True

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class OrderManager:
    """Persist intents and apply broker events with idempotent semantics."""

    def __init__(
        self,
        store: SQLiteOrderStore,
        broker: Optional[BrokerGateway] = None,
        clock: Callable[[], str] = utc_now,
        risk_manager: Any = None,
        account_provider: Any = None,
        submission_enabled: bool = True,
        submission_gate_reason: str = "",
    ) -> None:
        self.store = store
        self.broker = broker
        self.clock = clock
        self.risk_manager = risk_manager
        self.account_provider = account_provider
        self._lock = threading.RLock()
        self._submission_enabled = bool(submission_enabled)
        self._submission_gate_reason = str(submission_gate_reason or "")
        self._gate_revision = 0

    @property
    def submission_enabled(self) -> bool:
        with self._lock:
            return self._submission_enabled

    @property
    def gate_revision(self) -> int:
        with self._lock:
            return self._gate_revision

    @property
    def submission_gate_reason(self) -> str:
        with self._lock:
            return self._submission_gate_reason

    def set_gate(self, open_: bool, reason: str = "") -> bool:
        """Change the central order gate; closing it advances its safety epoch."""
        with self._lock:
            open_ = bool(open_)
            if not open_:
                self._gate_revision += 1
            self._submission_enabled = open_
            self._submission_gate_reason = "" if open_ else str(reason or "execution_gate_closed")
            return self._submission_enabled

    def begin_reconciliation(
        self, reason: str = "reconciliation_in_progress",
        expected_revision: Optional[int] = None,
    ) -> Optional[int]:
        """Close submissions and return an epoch for one reconciliation pass."""
        with self._lock:
            if expected_revision is not None and self._gate_revision != int(expected_revision):
                return None
            self._gate_revision += 1
            self._submission_enabled = False
            self._submission_gate_reason = str(reason)
            return self._gate_revision

    def open_gate_after_reconciliation(self, expected_revision: int) -> bool:
        """Open only if no disconnect/stop event occurred during reconciliation."""
        with self._lock:
            if self._gate_revision != int(expected_revision):
                return False
            self._submission_enabled = True
            self._submission_gate_reason = ""
            return True

    @staticmethod
    def _same_intent(record: OrderRecord, intent: OrderIntent) -> bool:
        return (
            record.account_id == intent.account_id
            and record.symbol == intent.symbol
            and record.side == str(intent.side)
            and record.quantity == intent.quantity
            and record.order_type == str(intent.order_type)
            and (record.limit_price is None and intent.limit_price is None
                 or record.limit_price is not None and intent.limit_price is not None
                 and abs(record.limit_price - intent.limit_price) < 1e-8)
            and record.signal_id == intent.signal_id
            and _stable_json(record.metadata) == _stable_json(dict(intent.metadata))
        )

    def get_or_create(self, intent: OrderIntent) -> Tuple[OrderRecord, bool]:
        with self._lock:
            existing = self.store.get(intent.client_order_id)
            if existing is not None:
                if not self._same_intent(existing, intent):
                    raise OrderConflictError(
                        f"client_order_id {intent.client_order_id} is already bound to another intent"
                    )
                return existing, False
            record = OrderRecord.from_intent(intent, now=self.clock())
            self.store.insert_with_event(record, "created", record.__dict__)
            return record, True

    def submit(self, intent: OrderIntent) -> OrderRecord:
        with self._lock:
            if not self._submission_enabled:
                reason = self._submission_gate_reason or "execution_gate_closed"
                raise ExecutionGateClosed(reason)
        prepare_intent = getattr(self.broker, "prepare_intent", None)
        if prepare_intent is not None:
            intent = prepare_intent(intent)
        record, is_new = self.get_or_create(intent)
        with self._lock:
            if not self._submission_enabled:
                reason = self._submission_gate_reason or "execution_gate_closed"
                raise ExecutionGateClosed(reason)
            if self.risk_manager is not None and record.status not in TERMINAL_STATUSES:
                if self.account_provider is None:
                    raise RuntimeError("risk_manager requires account_provider")
                account = self.account_provider.query_account(intent.account_id)
                positions = self.account_provider.query_positions(intent.account_id)
                try:
                    self.risk_manager.validate(intent, account, positions)
                except Exception as exc:
                    record.status = OrderStatus.REJECTED
                    record.reject_reason = f"risk_rejected:{exc}"
                    record.updated_at = self.clock()
                    self.store.save_with_event(record, "risk_rejected", {"reason": str(exc)})
                    return record
            if record.status in TERMINAL_STATUSES or record.broker_order_id:
                return record
            if not is_new and record.status != OrderStatus.PENDING:
                raise AmbiguousSubmissionError(
                    f"{record.client_order_id} has no confirmed broker order; reconcile before retrying"
                )
            if self.broker is None:
                raise OrderProtocolError("no broker gateway configured")
            # Persist an ambiguous/in-flight marker before the external call.
            # If the process stops at any point after this commit, a restart
            # cannot turn the same intent into another broker submission.
            record.status = OrderStatus.UNKNOWN
            record.updated_at = self.clock()
            self.store.save_with_event(record, "submission_started", {})
            try:
                result = self.broker.submit(intent)
            except Exception as exc:
                record.status = OrderStatus.UNKNOWN
                record.reject_reason = f"submission_exception:{type(exc).__name__}:{exc}"
                record.updated_at = self.clock()
                self.store.save_with_event(record, "submission_unknown", {"error": str(exc)})
                raise AmbiguousSubmissionError(record.reject_reason) from exc
            if not result.accepted:
                record.status = OrderStatus.REJECTED
                record.reject_reason = result.reason or "broker_rejected"
            else:
                if result.broker_order_id is None:
                    record.status = OrderStatus.UNKNOWN
                    record.reject_reason = "broker_accepted_without_order_id"
                    record.updated_at = self.clock()
                    self.store.save_with_event(record, "submission_unknown", {
                        "reason": record.reject_reason,
                    })
                    raise AmbiguousSubmissionError(record.reject_reason)
                # A fast gateway may deliver an order/fill callback before
                # submit() returns. Reload the committed state so its fill is
                # not overwritten by the submission acknowledgement.
                record = self.store.get(record.client_order_id) or record
                record.broker_order_id = str(result.broker_order_id)
                if result.broker_order_sys_id is not None:
                    record.broker_order_sys_id = str(result.broker_order_sys_id)
                if record.status == OrderStatus.UNKNOWN:
                    record.status = OrderStatus.PENDING
            record.updated_at = self.clock()
            self.store.save_with_event(record, "submitted", {
                "accepted": result.accepted,
                "broker_order_id": result.broker_order_id,
                "reason": result.reason,
            })
            return record

    def _resolve(self, client_order_id: Optional[str], broker_order_id: Optional[str], broker_order_sys_id: Optional[str]) -> Optional[OrderRecord]:
        record = self.store.get(client_order_id) if client_order_id else None
        if record is None and broker_order_id:
            record = self.store.find_by_broker_id(str(broker_order_id))
        if record is None and broker_order_sys_id:
            record = self.store.find_by_broker_sys_id(str(broker_order_sys_id))
        return record

    def apply_order_update(self, snapshot: BrokerOrderSnapshot) -> Optional[OrderRecord]:
        with self._lock:
            record = self._resolve(
                snapshot.client_order_id,
                snapshot.broker_order_id,
                snapshot.broker_order_sys_id,
            )
            if record is None:
                return None
            new_status = _normalise_status(snapshot.status)
            if snapshot.broker_order_id:
                record.broker_order_id = str(snapshot.broker_order_id)
            if snapshot.broker_order_sys_id:
                record.broker_order_sys_id = str(snapshot.broker_order_sys_id)
            if snapshot.filled_quantity < record.filled_quantity:
                # Broker snapshots can be stale; do not roll back fills.
                filled_quantity = record.filled_quantity
            else:
                filled_quantity = int(snapshot.filled_quantity)
            if filled_quantity > record.quantity:
                raise OrderProtocolError(f"broker reports overfill for {record.client_order_id}")
            # Some APIs publish a stale textual status while their traded
            # quantity is already ahead.  Quantity is the durable source of
            # truth for fill progress and prevents strategies from treating a
            # partially filled order as still pending.
            if filled_quantity >= record.quantity:
                new_status = OrderStatus.FILLED
            elif filled_quantity > 0 and new_status in {
                OrderStatus.PENDING,
                OrderStatus.CANCEL_REQUESTED,
                OrderStatus.UNKNOWN,
            }:
                new_status = OrderStatus.PARTIALLY_FILLED
            if new_status not in _ALLOWED_TRANSITIONS[record.status]:
                raise InvalidOrderState(
                    f"{record.client_order_id}: {record.status.value} -> {new_status.value}"
                )
            record.filled_quantity = filled_quantity
            if snapshot.avg_fill_price is not None:
                record.avg_fill_price = float(snapshot.avg_fill_price)
            record.status = new_status
            if snapshot.reject_reason:
                record.reject_reason = snapshot.reject_reason
            if snapshot.cancel_reason:
                record.cancel_reason = snapshot.cancel_reason
            record.updated_at = self.clock()
            self.store.save_with_event(record, "order_update", {
                "status": new_status.value,
                "filled_quantity": record.filled_quantity,
                "broker_order_id": record.broker_order_id,
            })
            return record

    def apply_fill(self, event: FillEvent) -> Optional[OrderRecord]:
        with self._lock:
            record = self._resolve(event.client_order_id, event.broker_order_id, event.broker_order_sys_id)
            if record is None:
                return None
            if event.symbol.upper() != record.symbol.upper():
                raise OrderProtocolError(
                    f"fill symbol {event.symbol} does not match {record.symbol} for {record.client_order_id}"
                )
            if self.store.has_fill(event.fill_id):
                return record
            new_filled = record.filled_quantity + event.quantity
            if new_filled > record.quantity:
                raise OrderProtocolError(f"broker reports overfill for {record.client_order_id}")
            old_value = (record.avg_fill_price or 0.0) * record.filled_quantity
            record.avg_fill_price = (old_value + event.price * event.quantity) / new_filled
            record.filled_quantity = new_filled
            record.status = OrderStatus.FILLED if new_filled == record.quantity else OrderStatus.PARTIALLY_FILLED
            if event.broker_order_id and not record.broker_order_id:
                record.broker_order_id = str(event.broker_order_id)
            if event.broker_order_sys_id and not record.broker_order_sys_id:
                record.broker_order_sys_id = str(event.broker_order_sys_id)
            record.updated_at = self.clock()
            if not self.store.add_fill_and_save(event, record.client_order_id, record, {
                "fill_id": event.fill_id,
                "quantity": event.quantity,
                "price": event.price,
                "filled_quantity": record.filled_quantity,
            }):
                # Another callback committed this fill while this thread was
                # calculating the aggregate.  Reloading avoids double counting.
                return self.store.get(record.client_order_id)
            return record

    def request_cancel(self, client_order_id: str) -> OrderRecord:
        with self._lock:
            record = self.store.get(client_order_id)
            if record is None:
                raise KeyError(client_order_id)
            if record.status in TERMINAL_STATUSES:
                return record
            if not record.broker_order_id:
                raise OrderProtocolError(f"{client_order_id} has no broker_order_id to cancel")
            if self.broker is None:
                raise OrderProtocolError("no broker gateway configured")
            accepted = self.broker.cancel(record)
            record = self.store.get(client_order_id) or record
            if accepted:
                if record.status not in TERMINAL_STATUSES:
                    record.status = OrderStatus.CANCEL_REQUESTED
            else:
                record.cancel_reason = "broker_cancel_rejected"
            record.updated_at = self.clock()
            self.store.save_with_event(record, "cancel_requested", {"accepted": accepted})
            return record

    def reconcile(
        self,
        snapshots: Iterable[BrokerOrderSnapshot],
        fills: Iterable[FillEvent] = (),
    ) -> Dict[str, Any]:
        """Apply broker snapshots and report unresolved local/broker orders.

        Reconciliation never submits or cancels automatically.  That boundary
        is important after a process restart or a network timeout.
        """

        for fill in fills:
            self.apply_fill(fill)

        matched: List[str] = []
        unknown_broker_orders: List[BrokerOrderSnapshot] = []
        for snapshot in snapshots:
            record = self.apply_order_update(snapshot)
            if record is None:
                unknown_broker_orders.append(snapshot)
            else:
                matched.append(record.client_order_id)
        open_records = self.store.list_open()
        seen = set(matched)
        missing_local_orders = [record.client_order_id for record in open_records if record.client_order_id not in seen]
        return {
            "matched_client_order_ids": matched,
            "unknown_broker_orders": unknown_broker_orders,
            "missing_local_order_ids": missing_local_orders,
            "requires_manual_review": bool(unknown_broker_orders or missing_local_orders),
        }
