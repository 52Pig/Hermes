"""Strict importer for account snapshots exported by a human.

This module is deliberately broker-neutral and read-only.  It accepts a JSON
export so the same snapshot can be archived and replayed, then validates it
before the inference pipeline sees the account state.  It never submits an
order and it does not infer missing broker fields from local order history.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from .account_protocol import AccountSnapshot, PositionSnapshot, validate_account_state


SHANGHAI = timezone(timedelta(hours=8))


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field}_must_be_iso_datetime")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field}_must_be_iso_datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field}_must_include_timezone")
    return parsed.astimezone(timezone.utc)


def _number(value: Any, field: str, *, minimum: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}_must_be_number") from exc
    if result != result or result in (float("inf"), float("-inf")) or result < minimum:
        raise ValueError(f"{field}_must_be_finite_and_nonnegative")
    return result


def load_manual_account_snapshot(
    path: str | Path,
    *,
    expected_account_id: str,
    max_age_seconds: int = 24 * 60 * 60,
    now: datetime | None = None,
) -> AccountSnapshot:
    """Load and validate one manually exported account snapshot.

    ``as_of`` is the time shown by the official broker terminal, not the file
    modification time.  A stale or future-dated snapshot is rejected so an
    operator cannot accidentally run inference against yesterday's account.
    """
    if not expected_account_id:
        raise ValueError("expected_account_id_required")
    if max_age_seconds <= 0:
        raise ValueError("max_age_seconds_must_be_positive")
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("account_snapshot_json_unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("account_snapshot_must_be_object")
    account_id = str(payload.get("account_id", ""))
    if account_id != str(expected_account_id):
        raise ValueError("account_snapshot_mismatch")
    currency = str(payload.get("currency", "CNY"))
    if currency != "CNY":
        raise ValueError("unsupported_account_currency")
    as_of = _timestamp(payload.get("as_of"), "as_of")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age = (current - as_of).total_seconds()
    if age < -60:
        raise ValueError("account_snapshot_from_future")
    if age > max_age_seconds:
        raise ValueError("account_snapshot_stale")

    raw_positions = payload.get("positions", [])
    if not isinstance(raw_positions, list):
        raise ValueError("positions_must_be_list")
    positions = []
    for row in raw_positions:
        if not isinstance(row, Mapping):
            raise ValueError("position_must_be_object")
        symbol = str(row.get("symbol", "")).upper()
        if not symbol:
            raise ValueError("position_symbol_required")
        positions.append(PositionSnapshot(
            account_id=account_id,
            symbol=symbol,
            quantity=_number(row.get("quantity"), f"{symbol}.quantity"),
            available_quantity=_number(row.get("available_quantity"), f"{symbol}.available_quantity"),
            average_price=_number(row.get("average_price"), f"{symbol}.average_price"),
            market_value=_number(row.get("market_value", 0.0), f"{symbol}.market_value"),
            side=str(row.get("side", "LONG")).upper(),
            today_quantity=_number(row.get("today_quantity", 0.0), f"{symbol}.today_quantity"),
            yesterday_quantity=_number(row.get("yesterday_quantity", 0.0), f"{symbol}.yesterday_quantity"),
            metadata={"source": "manual_broker_export"},
        ))
    account = AccountSnapshot(
        account_id=account_id,
        currency=currency,
        cash=_number(payload.get("cash"), "cash"),
        available_cash=_number(payload.get("available_cash"), "available_cash"),
        equity=_number(payload.get("equity"), "equity", minimum=0.01),
        margin_used=_number(payload.get("margin_used", 0.0), "margin_used"),
        frozen_cash=_number(payload.get("frozen_cash", 0.0), "frozen_cash"),
        positions=tuple(positions),
        as_of=as_of.isoformat(),
        metadata={
            "source": "manual_broker_export",
            "source_path": str(source),
            "loaded_at": current.isoformat(),
        },
    )
    validate_account_state(account, positions, expected_account_id)
    if account.available_cash > account.cash:
        raise ValueError("available_cash_exceeds_cash")
    if account.frozen_cash > account.cash:
        raise ValueError("frozen_cash_exceeds_cash")
    return account


def write_manual_snapshot_template(path: str | Path, account_id: str) -> None:
    """Write a blank operator template without contacting a broker."""
    if not account_id:
        raise ValueError("account_id_required")
    payload = {
        "account_id": account_id,
        "currency": "CNY",
        "as_of": "2026-01-01T15:00:00+08:00",
        "cash": 100000.0,
        "available_cash": 100000.0,
        "equity": 100000.0,
        "frozen_cash": 0.0,
        "margin_used": 0.0,
        "positions": [],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
