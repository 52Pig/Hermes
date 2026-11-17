"""Read-only portfolio shadow evaluation with durable cycle auditing.

The adapter stops at broker-neutral intents. A SQLite audit store makes a
completed cycle idempotent across process restarts and records attempts to
reuse a cycle id with changed inputs. It never submits, reserves cash, or
changes durable order state.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .order_protocol import OrderIntent
from .portfolio_pipeline import DecisionCycle, PortfolioDecisionEngine


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _stable_hash(cycle: DecisionCycle) -> str:
    payload = {
        "cycle_id": cycle.cycle_id,
        "as_of": cycle.as_of,
        "account": cycle.account.__dict__,
        "positions": [position.__dict__ for position in cycle.positions],
        "market": cycle.market,
    }
    try:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("decision cycle contains non-serializable shadow inputs") from exc
    return hashlib.sha256(encoded).hexdigest()


class SQLiteShadowAuditStore:
    """Durable, append-audited decision records for shadow cycles.

    The cycle table holds the first input hash and latest decision. The event
    table preserves every evaluation, repeat, and changed-input attempt so the
    history remains inspectable even when a failed evaluation is retried.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = str(path)
        if not self.path:
            raise ValueError("shadow audit path is required")
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path, timeout=30.0, check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout=30000")
        if self.path != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS shadow_cycles (
                    cycle_id TEXT PRIMARY KEY,
                    input_hash TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS shadow_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_shadow_events_cycle
                    ON shadow_events (cycle_id, event_id);
                """
            )
            self._connection.execute("PRAGMA user_version = 1")

    @staticmethod
    def _json(value: Mapping[str, Any]) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

    def get_cycle(self, cycle_id: str) -> Mapping[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT cycle_id, input_hash, decision_json FROM shadow_cycles WHERE cycle_id = ?",
                (cycle_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "cycle_id": row["cycle_id"],
            "input_hash": row["input_hash"],
            "decision": json.loads(row["decision_json"]),
        }

    def record_event(
        self, cycle_id: str, event_type: str, input_hash: str,
        payload: Mapping[str, Any],
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO shadow_events
                   (cycle_id, event_type, input_hash, event_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (cycle_id, event_type, input_hash, self._json(payload), _utc_now()),
            )

    def save_decision(
        self, cycle_id: str, input_hash: str, decision: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Save an attempt or return the existing winner for a concurrent cycle."""
        encoded = self._json(decision)
        now = _utc_now()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT input_hash, decision_json FROM shadow_cycles WHERE cycle_id = ?",
                (cycle_id,),
            ).fetchone()
            if row is None:
                self._connection.execute(
                    """INSERT INTO shadow_cycles
                       (cycle_id, input_hash, decision_json, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (cycle_id, input_hash, encoded, now, now),
                )
                stored = dict(decision)
                event_type = "evaluation_passed" if decision["status"] == "PASS" else "evaluation_failed"
                repeated = False
                changed = False
            elif row["input_hash"] != input_hash:
                stored = {
                    "input_hash": row["input_hash"],
                    "decision": json.loads(row["decision_json"]),
                }
                event_type = "cycle_input_changed"
                repeated = False
                changed = True
            else:
                # Same-input failures are retryable; successful decisions stay
                # immutable so concurrent or repeated evaluation cannot yield
                # a different shadow order set for the same cycle.
                previous = json.loads(row["decision_json"])
                if previous["status"] == "PASS":
                    stored = previous
                    event_type = "cycle_repeated"
                    repeated = True
                else:
                    self._connection.execute(
                        "UPDATE shadow_cycles SET decision_json = ?, updated_at = ? WHERE cycle_id = ?",
                        (encoded, now, cycle_id),
                    )
                    stored = dict(decision)
                    event_type = "evaluation_passed" if decision["status"] == "PASS" else "evaluation_failed"
                    repeated = False
                changed = False

            self._connection.execute(
                """INSERT INTO shadow_events
                   (cycle_id, event_type, input_hash, event_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (cycle_id, event_type, input_hash, self._json({"decision": decision}), now),
            )
        return {"decision": stored, "repeated": repeated, "changed": changed}

    def close(self) -> None:
        with self._lock:
            self._connection.close()


@dataclass(frozen=True)
class ShadowDecision:
    cycle_id: str
    input_hash: str
    order_count: int
    orders: tuple[Any, ...]
    status: str = "PASS"
    failure_code: str = ""
    failure_message: str = ""
    repeated: bool = False


def _encode_decision(decision: ShadowDecision) -> dict[str, Any]:
    encoded_orders = []
    for order in decision.orders:
        if not isinstance(order, OrderIntent):
            raise TypeError("shadow engine returned an unsupported order type")
        encoded_orders.append(order.to_dict())
    return {
        "input_hash": decision.input_hash,
        "order_count": decision.order_count,
        "orders": encoded_orders,
        "status": decision.status,
        "failure_code": decision.failure_code,
        "failure_message": decision.failure_message,
    }


def _decode_decision(cycle_id: str, payload: Mapping[str, Any], *, repeated: bool) -> ShadowDecision:
    order_fields = {item.name for item in fields(OrderIntent)}
    orders = tuple(
        OrderIntent(**{key: value for key, value in order.items() if key in order_fields})
        for order in payload.get("orders", ())
    )
    return ShadowDecision(
        cycle_id=cycle_id,
        input_hash=str(payload["input_hash"]),
        order_count=int(payload["order_count"]),
        orders=orders,
        status=str(payload["status"]),
        failure_code=str(payload.get("failure_code", "")),
        failure_message=str(payload.get("failure_message", "")),
        repeated=repeated,
    )


class PortfolioShadowRunner:
    """Build auditable decisions for observation and never submit them."""

    def __init__(
        self, engine: PortfolioDecisionEngine, *, audit_path: str | os.PathLike[str],
    ) -> None:
        self.engine = engine
        self.audit_store = SQLiteShadowAuditStore(audit_path)

    async def evaluate(self, cycle: DecisionCycle) -> ShadowDecision:
        input_hash = _stable_hash(cycle)
        previous = self.audit_store.get_cycle(cycle.cycle_id)
        if previous is not None:
            if previous["input_hash"] != input_hash:
                self.audit_store.record_event(
                    cycle.cycle_id, "cycle_input_changed", input_hash,
                    {"stored_input_hash": previous["input_hash"]},
                )
                return ShadowDecision(
                    cycle_id=cycle.cycle_id, input_hash=input_hash, order_count=0,
                    orders=(), status="FAIL", failure_code="cycle_input_changed",
                    failure_message="cycle_id was reused with a different input snapshot",
                )
            prior_decision = previous["decision"]
            if prior_decision["status"] == "PASS":
                self.audit_store.record_event(
                    cycle.cycle_id, "cycle_repeated", input_hash,
                    {"decision_status": prior_decision["status"]},
                )
                return _decode_decision(cycle.cycle_id, prior_decision, repeated=True)

        try:
            orders = tuple(await self.engine.build_orders(cycle))
            decision = ShadowDecision(
                cycle_id=cycle.cycle_id, input_hash=input_hash,
                order_count=len(orders), orders=orders,
            )
        except Exception as exc:  # shadow failures remain structured and durable
            decision = ShadowDecision(
                cycle_id=cycle.cycle_id, input_hash=input_hash, order_count=0,
                orders=(), status="FAIL", failure_code=type(exc).__name__,
                failure_message=str(exc),
            )

        outcome = self.audit_store.save_decision(
            cycle.cycle_id, input_hash, _encode_decision(decision),
        )
        if outcome["changed"]:
            stored = outcome["decision"]
            # Another runner won the first-write race for this cycle id.
            return ShadowDecision(
                cycle_id=cycle.cycle_id, input_hash=input_hash, order_count=0,
                orders=(), status="FAIL", failure_code="cycle_input_changed",
                failure_message="cycle_id was reused with a different input snapshot",
            )
        stored = outcome["decision"]
        if outcome["repeated"]:
            return _decode_decision(cycle.cycle_id, stored, repeated=True)
        return decision

    def close(self) -> None:
        self.audit_store.close()


def shadow_summary(decision: ShadowDecision) -> Mapping[str, Any]:
    """Return a JSON-safe operational record for logs/monitoring."""
    return {
        "cycle_id": decision.cycle_id,
        "input_hash": decision.input_hash,
        "order_count": decision.order_count,
        "status": decision.status,
        "failure_code": decision.failure_code,
        "failure_message": decision.failure_message,
        "repeated": decision.repeated,
        "orders": [
            {
                "client_order_id": getattr(order, "client_order_id", ""),
                "symbol": getattr(order, "symbol", ""),
                "side": getattr(order, "side", ""),
                "quantity": getattr(order, "quantity", 0),
                "limit_price": getattr(order, "limit_price", None),
            }
            for order in decision.orders
        ],
    }
