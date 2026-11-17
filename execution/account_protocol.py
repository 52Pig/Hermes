"""Broker-neutral account, position and risk contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Protocol, Sequence


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class PositionSnapshot:
    account_id: str
    symbol: str
    quantity: float
    available_quantity: float
    average_price: float
    market_value: float = 0.0
    side: str = "LONG"
    today_quantity: float = 0.0
    yesterday_quantity: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AccountSnapshot:
    account_id: str
    currency: str
    cash: float
    available_cash: float
    equity: float
    margin_used: float = 0.0
    frozen_cash: float = 0.0
    positions: Sequence[PositionSnapshot] = ()
    as_of: str = field(default_factory=utc_timestamp)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class AccountGateway(Protocol):
    def query_account(self, account_id: str) -> AccountSnapshot:
        ...

    def query_positions(self, account_id: str) -> Sequence[PositionSnapshot]:
        ...


def validate_account_state(
    account: AccountSnapshot,
    positions: Sequence[PositionSnapshot],
    expected_account_id: str,
) -> None:
    """Reject incomplete or internally inconsistent broker snapshots."""

    if str(account.account_id) != str(expected_account_id):
        raise ValueError("account_snapshot_mismatch")
    if account.equity <= 0:
        raise ValueError("non_positive_account_equity")
    if account.available_cash < 0 or account.cash < 0:
        raise ValueError("negative_account_cash")
    seen = set()
    for position in positions:
        if str(position.account_id) != str(expected_account_id):
            raise ValueError(f"position_account_mismatch:{position.symbol}")
        if not position.symbol or position.quantity < 0 or position.available_quantity < 0:
            raise ValueError(f"invalid_position_snapshot:{position.symbol}")
        if position.available_quantity > position.quantity:
            raise ValueError(f"available_exceeds_position:{position.symbol}")
        key = (position.symbol.upper(), position.side.upper())
        if key in seen:
            raise ValueError(f"duplicate_position_snapshot:{position.symbol}")
        seen.add(key)
