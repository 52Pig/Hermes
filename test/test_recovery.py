import asyncio

from execution.account_protocol import AccountSnapshot
from execution.order_protocol import OrderManager, SQLiteOrderStore, SubmissionResult
from execution.recovery import AccountGateController, AccountRecoveryCoordinator, QmtRecoverySupervisor
from execution.xtquant_gateway import XtQuantCallbackBridge


class Broker:
    def submit(self, intent):
        return SubmissionResult(accepted=True, broker_order_id="B-1")

    def cancel(self, record):
        return True


class HealthyRouter:
    def __init__(self, account_id="ACC"):
        self.account_id = account_id
        self.last_query_errors = {}
        self.on_query_orders = None

    def clear_query_errors(self):
        self.last_query_errors.clear()

    def query_orders(self):
        if self.on_query_orders:
            self.on_query_orders()
        return []

    def query_fills(self):
        return []

    def query_account(self, account_id):
        return AccountSnapshot(account_id, "CNY", 10000, 9000, 12000)

    def query_positions(self, account_id):
        return []


def closed_manager(tmp_path):
    return OrderManager(
        SQLiteOrderStore(str(tmp_path / "recovery.sqlite3")),
        broker=Broker(), submission_enabled=False,
        submission_gate_reason="startup_reconciliation_required",
    )


def test_recovery_coordinator_opens_only_after_clean_account_reconciliation(tmp_path):
    manager = closed_manager(tmp_path)
    router = HealthyRouter()
    report = AccountRecoveryCoordinator("ACC", manager, router).reconcile_and_open()

    assert manager.submission_enabled
    assert report["submission_gate_open"] is True
    assert report["position_count"] == 0
    assert not report["requires_manual_review"]


def test_recovery_coordinator_stays_closed_if_disconnect_occurs_during_checks(tmp_path):
    manager = closed_manager(tmp_path)
    router = HealthyRouter()
    router.on_query_orders = lambda: manager.set_gate(False, "qmt_disconnected")

    report = AccountRecoveryCoordinator("ACC", manager, router).reconcile_and_open()

    assert not manager.submission_enabled
    assert report["requires_manual_review"]
    assert report["startup_validation_error"] == "connection_lost_during_reconciliation"


def test_recovery_coordinator_fails_closed_on_channel_query_error(tmp_path):
    manager = closed_manager(tmp_path)
    class BrokenRouter(HealthyRouter):
        def query_orders(self):
            self.last_query_errors["ctp"] = "ConnectionError: disconnected"
            return []

    router = BrokenRouter()
    report = AccountRecoveryCoordinator("ACC", manager, router).reconcile_and_open()

    assert not manager.submission_enabled
    assert report["requires_manual_review"]
    assert report["gateway_query_errors"]["ctp"].startswith("ConnectionError")


def test_qmt_supervisor_reconnects_subscribes_then_reconciles(tmp_path):
    manager = closed_manager(tmp_path)
    router = HealthyRouter()
    coordinator = AccountRecoveryCoordinator("ACC", manager, router)
    bridge = XtQuantCallbackBridge(manager)

    class FakeTrader:
        connect_calls = 0
        subscribe_calls = 0

        def connect(self):
            self.connect_calls += 1
            return 0

        def subscribe(self, account):
            self.subscribe_calls += 1
            return 0

    class ManagedFacade:
        def __init__(self):
            self.gate = False

        def set_gate(self, value):
            self.gate = bool(value)

    trader = FakeTrader()
    facade = ManagedFacade()
    state = {
        "order_manager": manager,
        "callback_bridge": bridge,
        "raw_xt_trader": trader,
        "account": "ACC",
        "recovery_coordinator": coordinator,
        "subscription_ready": False,
        "xt_trader": facade,
    }

    opened = asyncio.run(QmtRecoverySupervisor(state).recover_once())

    assert opened
    assert manager.submission_enabled
    assert facade.gate
    assert trader.connect_calls == 1
    assert trader.subscribe_calls == 1
    assert state["managed_order_gate_open"]


def test_qmt_supervisor_keeps_gate_closed_when_subscribe_fails(tmp_path):
    manager = closed_manager(tmp_path)
    bridge = XtQuantCallbackBridge(manager)

    class FakeTrader:
        def connect(self):
            return 0

        def subscribe(self, account):
            return -1

    state = {
        "order_manager": manager,
        "callback_bridge": bridge,
        "raw_xt_trader": FakeTrader(),
        "account": "ACC",
        "recovery_coordinator": AccountRecoveryCoordinator("ACC", manager, HealthyRouter()),
        "subscription_ready": False,
    }
    opened = asyncio.run(QmtRecoverySupervisor(state).recover_once())

    assert not opened
    assert not manager.submission_enabled
    assert manager.submission_gate_reason == "qmt_subscribe_failed:-1"


def test_account_gate_controller_keeps_runtime_state_consistent(tmp_path):
    manager = closed_manager(tmp_path)

    class Facade:
        gate = True

        def set_gate(self, value):
            self.gate = bool(value)

    state = {"order_manager": manager, "xt_trader": Facade(), "managed_order_gate_open": True}
    controller = AccountGateController(state)
    controller.set_gate(False, reason="qmt_disconnected")
    assert not manager.submission_enabled
    assert not state["managed_order_gate_open"]
    assert state["gate_reason"] == "qmt_disconnected"
    assert not state["xt_trader"].gate

    # An account-state controller cannot bypass the coordinator's durable gate.
    assert not controller.set_gate(True)
    assert not manager.submission_enabled

    revision = manager.begin_reconciliation()
    assert manager.open_gate_after_reconciliation(revision)
    controller.set_gate(True)
    assert manager.submission_enabled
    assert state["managed_order_gate_open"]


def test_qmt_supervisor_blocks_manual_review_until_explicit_reset(tmp_path):
    manager = closed_manager(tmp_path)
    bridge = XtQuantCallbackBridge(manager)

    class UnknownOrderRouter(HealthyRouter):
        def query_orders(self):
            return [{"broker_order_id": "BROKER-UNKNOWN", "status": "submitted"}]

    state = {
        "order_manager": manager,
        "callback_bridge": bridge,
        "raw_xt_trader": type("Trader", (), {
            "connect": lambda self: 0,
            "subscribe": lambda self, account: 0,
        })(),
        "account": "ACC",
        "recovery_coordinator": AccountRecoveryCoordinator(
            "ACC", manager, UnknownOrderRouter()
        ),
        "subscription_ready": False,
    }
    supervisor = QmtRecoverySupervisor(state)

    assert not asyncio.run(supervisor.recover_once())
    assert supervisor.blocked
    assert state["recovery_blocked"]
    assert state["order_reconciliation"]["recovery_blocked"]
    assert not manager.submission_enabled

    # A blocked supervisor does not repeatedly reconnect or mutate state.
    assert not asyncio.run(supervisor.recover_once())
    assert supervisor.blocked

    supervisor.reset_recovery()
    assert not supervisor.blocked
    assert not state.get("recovery_blocked", False)
    assert state["subscription_ready"] is False
    assert manager.submission_gate_reason == "manual_review_reset_requires_reconciliation"
    snapshot = supervisor.status()
    assert snapshot["blocked"] is False
    assert snapshot["submission_enabled"] is False
    assert snapshot["subscription_ready"] is False


def test_qmt_supervisor_does_not_stick_on_transient_disconnect_report(tmp_path):
    manager = closed_manager(tmp_path)
    bridge = XtQuantCallbackBridge(manager)
    router = HealthyRouter()
    router.on_query_orders = lambda: manager.set_gate(False, "qmt_disconnected")
    state = {
        "order_manager": manager,
        "callback_bridge": bridge,
        "raw_xt_trader": type("Trader", (), {
            "connect": lambda self: 0,
            "subscribe": lambda self, account: 0,
        })(),
        "account": "ACC",
        "recovery_coordinator": AccountRecoveryCoordinator("ACC", manager, router),
        "subscription_ready": False,
    }
    supervisor = QmtRecoverySupervisor(state)

    assert not asyncio.run(supervisor.recover_once())
    assert not supervisor.blocked
