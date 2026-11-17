"""Central pre-trade risk checks shared by every execution channel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .account_protocol import AccountSnapshot, PositionSnapshot
from .order_protocol import OrderIntent, OrderSide, OrderType


class RiskRejected(RuntimeError):
    """Raised when an intent violates a centrally enforced risk limit."""


@dataclass(frozen=True)
class RiskLimits:
    max_order_equity_fraction: float = 0.25
    max_gross_equity_fraction: float = 1.0
    max_margin_equity_fraction: float = 0.40
    max_order_quantity: int = 1_000_000
    fee_buffer_fraction: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "max_order_equity_fraction",
            "max_gross_equity_fraction",
            "max_margin_equity_fraction",
            "fee_buffer_fraction",
        ):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.max_order_quantity <= 0:
            raise ValueError("max_order_quantity must be positive")


class RiskManager:
    def __init__(self, limits: Optional[RiskLimits] = None) -> None:
        self.limits = limits or RiskLimits()

    @staticmethod
    def _price(intent: OrderIntent) -> float:
        price = intent.limit_price or intent.metadata.get("reference_price")
        try:
            value = float(price)
        except (TypeError, ValueError):
            value = 0.0
        if value <= 0:
            raise RiskRejected("missing_positive_reference_price")
        return value

    @staticmethod
    def _gross_exposure(positions: Iterable[PositionSnapshot]) -> float:
        return sum(abs(float(position.market_value)) for position in positions)

    def validate(
        self,
        intent: OrderIntent,
        account: Optional[AccountSnapshot],
        positions: Iterable[PositionSnapshot],
    ) -> None:
        if account is None:
            raise RiskRejected("account_snapshot_unavailable")
        if account.account_id != intent.account_id:
            raise RiskRejected("account_snapshot_mismatch")
        if account.equity <= 0:
            raise RiskRejected("non_positive_account_equity")
        if intent.quantity > self.limits.max_order_quantity:
            raise RiskRejected("max_order_quantity_exceeded")

        price = self._price(intent)
        asset_class = str(intent.metadata.get("asset_class", "EQUITY")).upper()
        is_future = asset_class in {"FUTURE", "FUTURES"}
        multiplier = float(intent.metadata.get("contract_multiplier", 1.0))
        if multiplier <= 0:
            raise RiskRejected("invalid_contract_multiplier")
        notional = intent.quantity * price * multiplier
        positions = tuple(positions)
        current_gross = self._gross_exposure(positions)
        is_close = str(intent.metadata.get("position_effect", "")).upper() in {
            "CLOSE", "CLOSETODAY", "CLOSEYESTERDAY", "CLOSE_TODAY", "CLOSE_YESTERDAY"
        }
        if is_future:
            margin_rate = float(intent.metadata.get("margin_rate", 0) or 0)
            if margin_rate <= 0:
                raise RiskRejected("missing_positive_futures_margin_rate")
            required_margin = notional * margin_rate
            if intent.side == OrderSide.BUY.value and not is_close:
                if required_margin > account.equity * self.limits.max_order_equity_fraction:
                    raise RiskRejected("max_single_order_exposure_exceeded")
                if account.margin_used + required_margin > account.equity * self.limits.max_margin_equity_fraction:
                    raise RiskRejected("max_margin_exposure_exceeded")
                if required_margin > account.available_cash:
                    raise RiskRejected("insufficient_available_margin")
        elif intent.side == OrderSide.BUY.value and not is_close:
            if notional > account.equity * self.limits.max_order_equity_fraction:
                raise RiskRejected("max_single_order_exposure_exceeded")
            if notional * (1 + self.limits.fee_buffer_fraction) > account.available_cash:
                raise RiskRejected("insufficient_available_cash")
            if (current_gross + notional) > account.equity * self.limits.max_gross_equity_fraction:
                raise RiskRejected("max_gross_exposure_exceeded")

        if intent.side == OrderSide.SELL.value and not is_close:
            available = sum(
                max(0.0, position.available_quantity)
                for position in positions
                if position.symbol.upper() == intent.symbol.upper()
            )
            if intent.quantity > available:
                raise RiskRejected("sell_quantity_exceeds_available_position")
