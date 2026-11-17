import pytest

from execution.multi_gateway import CtpGateway, GatewayRouter, ManualApprovalGateway, PaperGateway, PTradeGateway
from execution.account_protocol import AccountSnapshot, PositionSnapshot, validate_account_state
from execution.order_protocol import OrderIntent, OrderManager, OrderRecord, OrderStatus, SQLiteOrderStore, SubmissionResult


class FakeClient:
    def __init__(self):
        self.requests = []
        self.cancelled = []

    def submit_order(self, request):
        self.requests.append(request)
        return {"accepted": True, "order_id": "ORD-1"}

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return True

    def query_orders(self, cancelable_only=False):
        return [{
            "order_id": "ORD-1", "client_order_id": "CLIENT-1", "symbol": "IF2612.CFFEX",
            "status": "PARTIAL", "filled_quantity": 1, "avg_fill_price": 4000,
        }]


def test_router_routes_to_configured_channel_and_normalizes_order_query():
    client = FakeClient()
    router = GatewayRouter({"ptrade": PTradeGateway(client)}, default_channel="ptrade")
    intent = OrderIntent(
        account_id="A1", strategy_name="equity", symbol="000001.SZ", side="BUY",
        quantity=100, order_type="LIMIT", limit_price=10, signal_id="sig-1",
        metadata={"execution_channel": "ptrade"},
    )

    result = router.submit(intent)
    snapshots = list(router.query_orders())

    assert result == SubmissionResult(accepted=True, broker_order_id="ORD-1")
    assert router.prepare_intent(intent).metadata["execution_channel"] == "ptrade"
    assert client.requests[0]["client_order_id"] == intent.client_order_id
    assert snapshots[0].status is OrderStatus.PARTIALLY_FILLED
    assert snapshots[0].client_order_id == "CLIENT-1"


def test_gateway_normalizes_ctp_order_and_trade_field_names_and_skips_empty_trade():
    class CtpLikeClient(FakeClient):
        def query_orders(self, cancelable_only=False):
            return [{"OrderRef": "CO-1", "OrderSysID": "SYS-1", "InstrumentID": "IF2612",
                     "OrderStatus": "1", "VolumeTraded": 2, "Price": 4001}]

        def query_fills(self):
            return [
                {"TradeID": "EMPTY", "OrderRef": "CO-1", "InstrumentID": "IF2612", "Volume": 0, "Price": 4000},
                {"TradeID": "T-1", "OrderRef": "CO-1", "InstrumentID": "IF2612", "Volume": 2, "Price": 4001},
            ]

    gateway = PTradeGateway(CtpLikeClient())
    order = list(gateway.query_orders())[0]
    fills = list(gateway.query_fills())
    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert order.broker_order_sys_id == "SYS-1"
    assert fills[0].fill_id == "T-1"
    assert fills[0].quantity == 2


def test_ctp_requires_futures_open_close_metadata():
    gateway = CtpGateway(FakeClient())
    intent = OrderIntent(
        account_id="F1", strategy_name="futures", symbol="IF2612.CFFEX", side="BUY",
        quantity=1, order_type="LIMIT", limit_price=4000, signal_id="f-1",
        metadata={"execution_channel": "ctp", "exchange": "CFFEX",
                  "position_effect": "OPEN", "hedge_flag": "SPECULATION"},
    )
    gateway.submit(intent)

    request = gateway.client.requests[0]
    assert request["exchange"] == "CFFEX"
    assert request["position_effect"] == "OPEN"
    assert request["hedge_flag"] == "SPECULATION"
    assert request["Direction"] == "Buy"
    assert request["CombOffsetFlag"] == "Open"
    assert request["CombHedgeFlag"] == "Speculation"
    assert request["OrderPriceType"] == "LimitPrice"
    assert request["InstrumentID"] == "IF2612"
    assert request["VolumeTotalOriginal"] == 1

    invalid = OrderIntent(
        account_id="F1", strategy_name="futures", symbol="IF2612.CFFEX", side="BUY",
        quantity=1, order_type="LIMIT", limit_price=4000, signal_id="f-2",
        metadata={"execution_channel": "ctp"},
    )
    with pytest.raises(ValueError, match="exchange"):
        gateway.submit(invalid)


def test_ctp_cancel_uses_order_action_fields_when_supported():
    class CtpClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.cancel_requests = []

        def cancel_order_ctp(self, request):
            self.cancel_requests.append(request)
            return {"success": True}

    client = CtpClient()
    gateway = CtpGateway(client)
    record = OrderRecord(
        client_order_id="CO-1", account_id="F1", strategy_name="futures",
        symbol="IF2612.CFFEX", side="BUY", quantity=1, order_type="LIMIT",
        limit_price=4000, broker_order_id="BROKER-1", broker_order_sys_id="SYS-1",
        metadata={"exchange": "CFFEX", "FrontID": 7, "SessionID": 8},
    )
    assert gateway.cancel(record)
    assert client.cancel_requests[0]["OrderSysID"] == "SYS-1"
    assert client.cancel_requests[0]["FrontID"] == 7
    assert client.cancel_requests[0]["SessionID"] == 8


def test_manual_gateway_fails_closed():
    intent = OrderIntent(
        account_id="A1", strategy_name="equity", symbol="000001.SZ", side="BUY",
        quantity=100, order_type="LIMIT", limit_price=10,
    )
    result = ManualApprovalGateway().submit(intent)
    assert not result.accepted
    assert result.reason == "manual_approval_required"


def test_paper_gateway_fill_and_cancel_are_connected_to_order_manager(tmp_path):
    paper = PaperGateway()
    manager = OrderManager(SQLiteOrderStore(str(tmp_path / "paper.sqlite3")), broker=paper)
    paper.bind_manager(manager)
    intent = OrderIntent(
        account_id="A1", strategy_name="paper", symbol="000001.SZ", side="BUY",
        quantity=200, order_type="LIMIT", limit_price=10, signal_id="paper-1",
        metadata={"execution_channel": "paper"},
    )
    record = manager.submit(intent)
    assert record.status is OrderStatus.PENDING
    assert paper.fill(record.broker_order_id, 100, 10.1) is OrderStatus.PARTIALLY_FILLED
    assert manager.store.get(record.client_order_id).filled_quantity == 100
    assert manager.request_cancel(record.client_order_id).status is OrderStatus.CANCELLED


def test_router_isolates_gateway_query_failure_and_records_health_error():
    class BrokenClient(FakeClient):
        def query_orders(self, cancelable_only=False):
            raise ConnectionError("session down")

    healthy = PTradeGateway(FakeClient())
    broken = PTradeGateway(BrokenClient())
    router = GatewayRouter({"healthy": healthy, "broken": broken}, default_channel="healthy")
    snapshots = list(router.query_orders())
    assert len(snapshots) == 1
    assert "broken" in router.last_query_errors


def test_order_and_fill_query_errors_are_both_retained_for_startup_gate():
    class OrdersBroken(FakeClient):
        def query_orders(self, cancelable_only=False):
            raise ConnectionError("orders unavailable")

    class FillsBroken(FakeClient):
        def query_fills(self):
            raise ConnectionError("fills unavailable")

    router = GatewayRouter({
        "orders": PTradeGateway(OrdersBroken()),
        "fills": PTradeGateway(FillsBroken()),
    })
    router.clear_query_errors()
    assert len(router.query_orders()) == 1
    assert router.query_fills() == []
    assert set(router.last_query_errors) == {"orders", "fills"}


def test_router_query_iterables_are_materialized_before_return():
    class BrokenClient(FakeClient):
        def query_orders(self, cancelable_only=False):
            raise ConnectionError("session down")

    router = GatewayRouter({"broken": PTradeGateway(BrokenClient())})
    result = router.query_orders()
    assert isinstance(result, list)
    assert "broken" in router.last_query_errors


def test_account_snapshot_validation_checks_account_and_position_integrity():
    account = AccountSnapshot("A1", "CNY", 1000, 900, 1200)
    valid_position = PositionSnapshot("A1", "000001.SZ", 100, 50, 10)
    validate_account_state(account, [valid_position], "A1")

    with pytest.raises(ValueError, match="account_snapshot_mismatch"):
        validate_account_state(account, [], "A2")
    invalid_position = PositionSnapshot("A1", "000001.SZ", 100, 101, 10)
    with pytest.raises(ValueError, match="available_exceeds_position"):
        validate_account_state(account, [invalid_position], "A1")
