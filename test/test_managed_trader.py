from types import SimpleNamespace
import pytest

from execution.managed_trader import ManagedTrader
from execution.order_protocol import (
    ExecutionGateClosed, OrderIntent, OrderManager, OrderRecord, OrderStatus,
    SQLiteOrderStore, SubmissionResult,
)
from execution.account_protocol import AccountSnapshot, PositionSnapshot
from execution.xtquant_gateway import XtQuantCallbackBridge


class FakeBroker:
    def __init__(self):
        self.intents = []

    def submit(self, intent):
        self.intents.append(intent)
        return SubmissionResult(accepted=True, broker_order_id="101")

    def cancel(self, record):
        return True


class FakeManager:
    def __init__(self):
        self.broker = FakeBroker()
        self.records = {}

    def submit(self, intent):
        if intent.client_order_id not in self.records:
            self.records[intent.client_order_id] = OrderRecord.from_intent(intent)
            self.records[intent.client_order_id].broker_order_id = "101"
        return self.records[intent.client_order_id]


def test_legacy_order_is_translated_and_reused():
    manager = FakeManager()
    raw = SimpleNamespace(query_stock_asset=lambda account: None)
    trader = ManagedTrader(raw, "ACC", manager, gate_open=True)

    first = trader.order_stock("ACC", "000001.SZ", 23, 100, 11, 10.5, "legacy_strategy")
    second = trader.order_stock("ACC", "000001.SZ", 23, 100, 11, 10.5, "legacy_strategy")

    assert first == second == 101
    assert len(manager.records) == 1
    intent = next(iter(manager.records.values())).intent()
    assert intent.side == "BUY"
    assert intent.order_type == "LIMIT"
    assert intent.limit_price == 10.5


def test_closed_gate_rejects_legacy_submission():
    manager = FakeManager()
    trader = ManagedTrader(SimpleNamespace(), "ACC", manager, gate_open=False)
    assert trader.order_stock("ACC", "600000.SH", 24, 100, 11, 9.8) == -1
    assert not manager.records


def test_gate_can_be_closed_after_connection_loss():
    manager = FakeManager()
    trader = ManagedTrader(SimpleNamespace(), "ACC", manager, gate_open=True)
    assert trader.gate_open
    trader.set_gate(False)
    assert not trader.gate_open
    assert trader.order_stock("ACC", "600000.SH", 23, 100, 11, 9.8) == -1


def test_qmt_disconnect_closes_bound_managed_gate_until_reconciliation():
    manager = FakeManager()
    trader = ManagedTrader(SimpleNamespace(), "ACC", manager, gate_open=True)
    bridge = XtQuantCallbackBridge(manager)
    bridge.bind_gate_controller(trader)
    bridge.on_connected()
    assert bridge.connected and trader.gate_open
    bridge.on_disconnected()
    assert not bridge.connected
    assert bridge.last_disconnect_reason == "qmt_disconnected"
    assert not trader.gate_open


def test_connect_return_does_not_override_racing_disconnect_callback():
    bridge = XtQuantCallbackBridge(FakeManager())
    attempt_revision = bridge.connection_revision
    bridge.on_disconnected()
    assert not bridge.mark_connected_if_unchanged(attempt_revision)
    assert not bridge.connected


def test_disconnect_blocks_both_direct_and_legacy_order_paths(tmp_path):
    broker = FakeBroker()
    manager = OrderManager(SQLiteOrderStore(str(tmp_path / "live.sqlite3")), broker=broker)
    trader = ManagedTrader(SimpleNamespace(), "ACC", manager, gate_open=True)
    bridge = XtQuantCallbackBridge(manager)
    bridge.bind_gate_controller(manager)
    bridge.bind_gate_controller(trader)

    bridge.on_disconnected()

    assert not manager.submission_enabled
    assert not trader.gate_open
    assert trader.order_stock("ACC", "600000.SH", 23, 100, 11, 9.8) == -1
    with pytest.raises(ExecutionGateClosed):
        manager.submit(OrderIntent(
            account_id="ACC", strategy_name="direct", symbol="600000.SH",
            side="BUY", quantity=100, order_type="LIMIT", limit_price=9.8,
            signal_id="disconnect-test",
        ))
    assert broker.intents == []


def test_market_order_has_no_limit_price():
    manager = FakeManager()
    trader = ManagedTrader(SimpleNamespace(), "ACC", manager, gate_open=True)
    trader.order_stock("ACC", "600000.SH", 24, 100, 5, 0, "legacy_strategy")
    intent = next(iter(manager.records.values())).intent()
    assert intent.side == "SELL"
    assert intent.order_type == "MARKET"
    assert intent.limit_price is None


def test_account_queries_keep_legacy_qmt_field_names():
    class Gateway:
        def query_account(self, account_id):
            return AccountSnapshot(
                account_id=account_id, currency="CNY", cash=100000,
                available_cash=90000, equity=120000,
            )

        def query_positions(self, account_id):
            return [PositionSnapshot(
                account_id=account_id, symbol="000001.SZ", quantity=200,
                available_quantity=100, average_price=10.0,
                market_value=2200, side="LONG",
            )]

        def query_orders(self, cancelable_only=False):
            return []

    trader = ManagedTrader(SimpleNamespace(), "ACC", FakeManager(),
                           gate_open=True, account_gateway=Gateway())
    asset = trader.query_stock_asset("ACC")
    positions = trader.query_stock_positions("ACC")
    assert asset.available_cash == 90000
    assert asset.total_asset == 120000
    assert positions[0].stock_code == "000001.SZ"
    assert positions[0].can_use_volume == 100
