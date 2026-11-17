import pytest

from execution.account_protocol import AccountSnapshot, PositionSnapshot
from execution.order_protocol import OrderIntent, OrderManager, OrderStatus, SQLiteOrderStore, SubmissionResult
from execution.risk import RiskLimits, RiskManager, RiskRejected


class Broker:
    def __init__(self):
        self.calls = 0

    def submit(self, intent):
        self.calls += 1
        return SubmissionResult(accepted=True, broker_order_id="R-1")

    def cancel(self, record):
        return True


class AccountProvider:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def query_account(self, account_id):
        return self.snapshot

    def query_positions(self, account_id):
        return self.snapshot.positions


def make_intent(**metadata):
    return OrderIntent(
        account_id="A1", strategy_name="risk", symbol="000001.SZ", side="BUY",
        quantity=100, order_type="LIMIT", limit_price=10, signal_id="risk-1",
        metadata=metadata,
    )


def test_risk_reject_is_persisted_and_broker_is_not_called(tmp_path):
    broker = Broker()
    account = AccountSnapshot("A1", "CNY", 10000, 50, 10000)
    manager = OrderManager(
        SQLiteOrderStore(str(tmp_path / "risk.sqlite3")), broker=broker,
        risk_manager=RiskManager(RiskLimits(max_order_equity_fraction=0.5)),
        account_provider=AccountProvider(account),
    )
    record = manager.submit(make_intent())

    assert record.status is OrderStatus.REJECTED
    assert "insufficient_available_cash" in record.reject_reason
    assert broker.calls == 0


def test_ctp_margin_limit_is_checked():
    account = AccountSnapshot("A1", "CNY", 100000, 100000, 100000, margin_used=39000)
    intent = OrderIntent(
        account_id="A1", strategy_name="futures", symbol="IF2612.CFFEX", side="BUY",
        quantity=1, order_type="LIMIT", limit_price=4000, signal_id="f-1",
        metadata={"asset_class": "FUTURES", "contract_multiplier": 300,
                  "margin_rate": 0.2, "position_effect": "OPEN"},
    )
    with pytest.raises(RiskRejected, match="max_margin_exposure_exceeded"):
        RiskManager(RiskLimits(
            max_order_equity_fraction=10.0, max_margin_equity_fraction=0.4
        )).validate(intent, account, ())
