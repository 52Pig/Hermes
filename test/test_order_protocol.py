import pytest

from execution.order_protocol import (
    AmbiguousSubmissionError,
    BrokerOrderSnapshot,
    FillEvent,
    ExecutionGateClosed,
    InvalidOrderState,
    OrderIntent,
    OrderManager,
    OrderStatus,
    SQLiteOrderStore,
    SubmissionResult,
)


class FakeBroker:
    def __init__(self, broker_order_id="B-1", raises=False):
        self.broker_order_id = broker_order_id
        self.raises = raises
        self.submit_calls = []
        self.cancel_calls = []

    def submit(self, intent):
        self.submit_calls.append(intent.client_order_id)
        if self.raises:
            raise TimeoutError("broker response lost")
        return SubmissionResult(accepted=True, broker_order_id=self.broker_order_id)

    def cancel(self, record):
        self.cancel_calls.append(record.client_order_id)
        return True


def intent(**overrides):
    values = {
        "account_id": "ACC-1",
        "strategy_name": "Top10V8V1",
        "symbol": "000001.SZ",
        "side": "BUY",
        "quantity": 100,
        "order_type": "LIMIT",
        "limit_price": 10.25,
        "signal_id": "20261025-rank-1",
        "signal_timestamp": "2026-10-25T06:50:00+00:00",
    }
    values.update(overrides)
    return OrderIntent(**values)


def manager(tmp_path, broker=None):
    return OrderManager(SQLiteOrderStore(str(tmp_path / "orders.sqlite3")), broker=broker)


def test_same_intent_is_idempotent_and_different_intent_conflicts(tmp_path):
    broker = FakeBroker()
    order_manager = manager(tmp_path, broker)
    first = order_manager.submit(intent())
    second = order_manager.submit(intent())

    assert first.client_order_id == second.client_order_id
    assert first.broker_order_id == "B-1"
    assert broker.submit_calls == [first.client_order_id]

    with pytest.raises(Exception):
        order_manager.submit(intent(quantity=200))
    assert broker.submit_calls == [first.client_order_id]


def test_duplicate_fill_is_consumed_once_and_average_price_is_weighted(tmp_path):
    order_manager = manager(tmp_path, FakeBroker())
    order = order_manager.submit(intent(quantity=200))
    fill = FillEvent(
        fill_id="F-1",
        broker_order_id=order.broker_order_id,
        symbol="000001.SZ",
        quantity=100,
        price=10.0,
        execution_timestamp="2026-10-25T01:30:00+00:00",
    )

    order_manager.apply_fill(fill)
    order_manager.apply_fill(fill)
    updated = order_manager.apply_fill(
        FillEvent(
            fill_id="F-2",
            broker_order_id=order.broker_order_id,
            symbol="000001.SZ",
            quantity=100,
            price=11.0,
            execution_timestamp="2026-10-25T01:31:00+00:00",
        )
    )

    assert updated.status is OrderStatus.FILLED
    assert updated.filled_quantity == 200
    assert updated.avg_fill_price == pytest.approx(10.5)


def test_unknown_submission_is_never_retried_without_reconciliation(tmp_path):
    broker = FakeBroker(raises=True)
    order_manager = manager(tmp_path, broker)

    with pytest.raises(AmbiguousSubmissionError):
        order_manager.submit(intent())
    record = order_manager.store.get(intent().client_order_id)
    assert record.status is OrderStatus.UNKNOWN
    assert broker.submit_calls == [record.client_order_id]

    with pytest.raises(AmbiguousSubmissionError):
        order_manager.submit(intent())
    assert broker.submit_calls == [record.client_order_id]


def test_reconciliation_binds_broker_snapshot_by_client_order_id(tmp_path):
    broker = FakeBroker()
    order_manager = manager(tmp_path, broker)
    order = order_manager.get_or_create(intent())[0]
    report = order_manager.reconcile(
        [
            BrokerOrderSnapshot(
                broker_order_id="B-77",
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                status="PENDING",
            )
        ]
    )

    restored = order_manager.store.get(order.client_order_id)
    assert restored.broker_order_id == "B-77"
    assert report["requires_manual_review"] is False
    assert report["matched_client_order_ids"] == [order.client_order_id]


def test_terminal_order_cannot_reopen(tmp_path):
    order_manager = manager(tmp_path, FakeBroker())
    order = order_manager.submit(intent())
    order_manager.apply_order_update(
        BrokerOrderSnapshot(
            broker_order_id=order.broker_order_id,
            symbol=order.symbol,
            status="REJECTED",
            reject_reason="risk",
        )
    )

    with pytest.raises(InvalidOrderState):
        order_manager.apply_order_update(
            BrokerOrderSnapshot(
                broker_order_id=order.broker_order_id,
                symbol=order.symbol,
                status="PENDING",
            )
        )


def test_state_and_fill_dedupe_survive_process_restart(tmp_path):
    broker = FakeBroker()
    first_manager = manager(tmp_path, broker)
    order = first_manager.submit(intent(quantity=200))
    first_manager.apply_fill(
        FillEvent(
            fill_id="F-restart",
            broker_order_id=order.broker_order_id,
            symbol=order.symbol,
            quantity=100,
            price=10.50,
            execution_timestamp="2026-10-25T01:30:00+00:00",
        )
    )
    first_manager.store.close()

    restarted_store = SQLiteOrderStore(str(tmp_path / "orders.sqlite3"))
    restarted = OrderManager(restarted_store, broker)
    restored = restarted_store.get(order.client_order_id)
    assert restored.status is OrderStatus.PARTIALLY_FILLED
    restarted.apply_fill(
        FillEvent(
            fill_id="F-restart",
            broker_order_id=order.broker_order_id,
            symbol=order.symbol,
            quantity=100,
            price=10.50,
            execution_timestamp="2026-10-25T01:30:00+00:00",
        )
    )
    assert restarted_store.get(order.client_order_id).filled_quantity == 100


def test_synchronous_fill_during_submit_is_not_overwritten(tmp_path):
    class CallbackBroker(FakeBroker):
        manager = None

        def submit(self, order_intent):
            result = super().submit(order_intent)
            self.manager.apply_fill(FillEvent(
                fill_id="FAST-FILL", client_order_id=order_intent.client_order_id,
                broker_order_id=result.broker_order_id, symbol=order_intent.symbol,
                quantity=100, price=10.25, execution_timestamp="2026-10-25T01:00:00Z",
            ))
            return result

    broker = CallbackBroker()
    order_manager = manager(tmp_path, broker)
    broker.manager = order_manager
    order = order_manager.submit(intent(quantity=200))

    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert order.filled_quantity == 100


def test_synchronous_cancel_callback_is_not_overwritten(tmp_path):
    class CallbackBroker(FakeBroker):
        manager = None
        order = None

        def cancel(self, record):
            self.manager.apply_order_update(BrokerOrderSnapshot(
                broker_order_id=record.broker_order_id, symbol=record.symbol,
                status=OrderStatus.CANCELLED,
            ))
            return True

    broker = CallbackBroker()
    order_manager = manager(tmp_path, broker)
    broker.manager = order_manager
    order = order_manager.submit(intent())

    cancelled = order_manager.request_cancel(order.client_order_id)
    assert cancelled.status is OrderStatus.CANCELLED


def test_order_snapshot_derives_fill_status_from_traded_quantity(tmp_path):
    order_manager = manager(tmp_path, FakeBroker())
    order = order_manager.submit(intent(quantity=200))

    partial = order_manager.apply_order_update(BrokerOrderSnapshot(
        broker_order_id=order.broker_order_id,
        symbol=order.symbol,
        status=OrderStatus.PENDING,
        filled_quantity=100,
        avg_fill_price=10.4,
    ))
    assert partial.status is OrderStatus.PARTIALLY_FILLED
    assert partial.filled_quantity == 100

    filled = order_manager.apply_order_update(BrokerOrderSnapshot(
        broker_order_id=order.broker_order_id,
        symbol=order.symbol,
        status=OrderStatus.PENDING,
        filled_quantity=200,
        avg_fill_price=10.5,
    ))
    assert filled.status is OrderStatus.FILLED
    assert filled.filled_quantity == 200


def test_central_gate_blocks_direct_submission_until_opened(tmp_path):
    broker = FakeBroker()
    order_manager = OrderManager(
        SQLiteOrderStore(str(tmp_path / "gated.sqlite3")), broker=broker,
        submission_enabled=False,
        submission_gate_reason="startup_reconciliation_required",
    )
    with pytest.raises(ExecutionGateClosed, match="startup_reconciliation_required"):
        order_manager.submit(intent())
    assert broker.submit_calls == []
    assert order_manager.store.list_all() == []

    revision = order_manager.gate_revision
    assert order_manager.open_gate_after_reconciliation(revision)
    order = order_manager.submit(intent())
    assert order.broker_order_id == "B-1"
    assert broker.submit_calls == [order.client_order_id]


def test_disconnect_during_reconciliation_prevents_gate_reopen(tmp_path):
    from execution.xtquant_gateway import XtQuantCallbackBridge

    order_manager = OrderManager(
        SQLiteOrderStore(str(tmp_path / "disconnect.sqlite3")), broker=FakeBroker(),
        submission_enabled=False,
        submission_gate_reason="startup_reconciliation_required",
    )
    bridge = XtQuantCallbackBridge(order_manager)
    bridge.bind_gate_controller(order_manager)
    expected_revision = order_manager.gate_revision

    bridge.on_connected()
    bridge.on_disconnected()
    bridge.on_connected()

    assert bridge.connected
    assert not order_manager.submission_enabled
    assert not order_manager.open_gate_after_reconciliation(expected_revision)
    with pytest.raises(ExecutionGateClosed):
        order_manager.submit(intent())
