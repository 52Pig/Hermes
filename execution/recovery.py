"""Fail-closed account recovery and broker reconciliation orchestration."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict

from .account_protocol import validate_account_state


class AccountGateController:
    """Keep the durable manager, legacy facade, and account state in sync."""

    def __init__(self, account_state: Dict[str, Any]) -> None:
        self.account_state = account_state

    def set_gate(self, open_: bool, reason: str = "") -> bool:
        manager = self.account_state.get("order_manager")
        if not open_ and manager is not None and manager is not self:
            manager.set_gate(False, reason=reason)
        effective_open = bool(
            open_ and manager is not None
            and getattr(manager, "submission_enabled", False)
        )
        managed_trader = self.account_state.get("xt_trader")
        if managed_trader is not None and managed_trader is not self:
            setter = getattr(managed_trader, "set_gate", None)
            if callable(setter):
                setter(effective_open)
        self.account_state["managed_order_gate_open"] = effective_open
        if reason:
            self.account_state["gate_reason"] = str(reason)
        elif effective_open:
            self.account_state.pop("gate_reason", None)
        return self.account_state["managed_order_gate_open"]


class AccountRecoveryCoordinator:
    """Reconcile one account before allowing any strategy to submit orders."""

    def __init__(self, account_id: str, order_manager: Any, gateway_router: Any) -> None:
        self.account_id = str(account_id)
        self.order_manager = order_manager
        self.gateway_router = gateway_router
        self._lock = threading.Lock()

    def reconcile_and_open(self, expected_gate_revision: int | None = None) -> Dict[str, Any]:
        with self._lock:
            revision = self.order_manager.begin_reconciliation(
                expected_revision=expected_gate_revision,
            )
            if revision is None:
                return {
                    "matched_client_order_ids": [],
                    "unknown_broker_orders": [],
                    "missing_local_order_ids": [],
                    "requires_manual_review": True,
                    "startup_validation_error": "connection_lost_before_reconciliation",
                }
            try:
                clear_errors = getattr(self.gateway_router, "clear_query_errors", None)
                if clear_errors:
                    clear_errors()
                broker_orders = list(self.gateway_router.query_orders())
                broker_fills = list(self.gateway_router.query_fills())
                report = self.order_manager.reconcile(broker_orders, broker_fills)

                account = self.gateway_router.query_account(self.account_id)
                positions = tuple(self.gateway_router.query_positions(self.account_id))
                validate_account_state(account, positions, self.account_id)
                report["account_snapshot_as_of"] = account.as_of
                report["position_count"] = len(positions)

                query_errors = dict(getattr(self.gateway_router, "last_query_errors", {}) or {})
                if query_errors:
                    report["gateway_query_errors"] = query_errors
                    report["requires_manual_review"] = True

                if report.get("requires_manual_review", False):
                    self.order_manager.set_gate(False, "reconciliation_requires_manual_review")
                    return report

                if not self.order_manager.open_gate_after_reconciliation(revision):
                    report["requires_manual_review"] = True
                    report["startup_validation_error"] = "connection_lost_during_reconciliation"
                    return report
                report["submission_gate_open"] = True
                return report
            except Exception as exc:
                self.order_manager.set_gate(False, "reconciliation_failed")
                return {
                    "matched_client_order_ids": [],
                    "unknown_broker_orders": [],
                    "missing_local_order_ids": [],
                    "requires_manual_review": True,
                    "startup_validation_error": f"{type(exc).__name__}: {exc}",
                }


class QmtRecoverySupervisor:
    """Retry QMT connection and reopen trading only after a clean recheck."""

    def __init__(self, account_state: Dict[str, Any], *, retry_min_seconds: float = 2.0,
                 retry_max_seconds: float = 30.0) -> None:
        self.account_state = account_state
        self.retry_min_seconds = max(0.1, float(retry_min_seconds))
        self.retry_max_seconds = max(self.retry_min_seconds, float(retry_max_seconds))
        self.blocked = bool(account_state.get("recovery_blocked", False))
        self.blocked_reason = str(account_state.get("recovery_block_reason", ""))

    @staticmethod
    def _is_manual_review_report(report: Dict[str, Any]) -> bool:
        """Return true for durable state mismatches requiring an operator decision.

        A disconnect during the reconciliation window is transient and remains
        eligible for automatic retry. Unknown/missing orders, query errors, and
        account validation failures are deliberately sticky until reviewed.
        """
        if not report.get("requires_manual_review", False):
            return False
        if report.get("unknown_broker_orders") or report.get("missing_local_order_ids"):
            return True
        if report.get("gateway_query_errors"):
            return True
        error = str(report.get("startup_validation_error", ""))
        return error not in {
            "connection_lost_before_reconciliation",
            "connection_lost_during_reconciliation",
        }

    def _set_blocked(self, reason: str) -> None:
        self.blocked = True
        self.blocked_reason = str(reason or "manual_review_required")
        self.account_state["recovery_blocked"] = True
        self.account_state["recovery_block_reason"] = self.blocked_reason
        manager = self.account_state.get("order_manager")
        if manager is not None:
            manager.set_gate(False, self.blocked_reason)

    def reset_recovery(self) -> None:
        """Explicitly clear a manual-review block and require a fresh reconcile."""
        self.blocked = False
        self.blocked_reason = ""
        self.account_state.pop("recovery_blocked", None)
        self.account_state.pop("recovery_block_reason", None)
        self.account_state["subscription_ready"] = False
        manager = self.account_state.get("order_manager")
        if manager is not None:
            manager.set_gate(False, "manual_review_reset_requires_reconciliation")
        controller = self.account_state.get("gate_controller")
        if controller is not None:
            controller.set_gate(False, "manual_review_reset_requires_reconciliation")

    def status(self) -> Dict[str, Any]:
        """Return a side-effect-free operational snapshot for monitoring/UI."""
        manager = self.account_state.get("order_manager")
        bridge = self.account_state.get("callback_bridge")
        return {
            "blocked": bool(self.blocked),
            "blocked_reason": self.blocked_reason,
            "connected": bool(getattr(bridge, "connected", False)),
            "subscription_ready": bool(self.account_state.get("subscription_ready", False)),
            "submission_enabled": bool(
                getattr(manager, "submission_enabled", False)
            ),
            "gate_reason": getattr(manager, "submission_gate_reason", ""),
            "last_reconciliation": self.account_state.get("order_reconciliation"),
        }

    async def recover_once(self) -> bool:
        manager = self.account_state.get("order_manager")
        bridge = self.account_state.get("callback_bridge")
        trader = self.account_state.get("raw_xt_trader")
        account = self.account_state.get("account")
        coordinator = self.account_state.get("recovery_coordinator")
        if manager is None or bridge is None or trader is None or coordinator is None:
            return False
        if self.blocked:
            manager.set_gate(False, self.blocked_reason or "recovery_manual_review_required")
            return False
        if manager.submission_enabled:
            return True

        reconnecting = not bridge.connected
        expected_gate_revision = manager.gate_revision
        if reconnecting:
            expected_connection_revision = bridge.connection_revision
            connect_result = await asyncio.to_thread(trader.connect)
            if connect_result != 0:
                manager.set_gate(False, f"qmt_reconnect_failed:{connect_result}")
                return False
            # Some QMT builds do not emit on_connected for a reconnect call.
            # Treat a successful connect return as provisional health; any
            # later on_disconnected callback still advances the gate revision
            # and invalidates this reconciliation pass.
            if not bridge.mark_connected_if_unchanged(expected_connection_revision):
                if not bridge.connected:
                    manager.set_gate(False, "qmt_disconnected_during_reconnect")
                    return False

        if reconnecting or not self.account_state.get("subscription_ready", False):
            subscribe_result = await asyncio.to_thread(trader.subscribe, account)
            if subscribe_result != 0:
                self.account_state["subscription_ready"] = False
                manager.set_gate(False, f"qmt_subscribe_failed:{subscribe_result}")
                return False
            self.account_state["subscription_ready"] = True

        if not bridge.connected:
            manager.set_gate(False, "qmt_connection_not_confirmed")
            return False
        report = await asyncio.to_thread(
            coordinator.reconcile_and_open, expected_gate_revision
        )
        self.account_state["order_reconciliation"] = report
        if self._is_manual_review_report(report):
            reason = report.get("startup_validation_error") or "reconciliation_requires_manual_review"
            self._set_blocked(str(reason))
            report["recovery_blocked"] = True
            report["recovery_block_reason"] = self.blocked_reason
            self.account_state["order_reconciliation"] = report
            return False
        opened = bool(manager.submission_enabled)
        gate_controller = self.account_state.get("gate_controller")
        if gate_controller is not None:
            gate_controller.set_gate(
                opened,
                reason="" if opened else (manager.submission_gate_reason or "reconciliation_requires_manual_review"),
            )
        else:
            self.account_state["managed_order_gate_open"] = opened
            managed_trader = self.account_state.get("xt_trader")
            if callable(getattr(managed_trader, "set_gate", None)):
                managed_trader.set_gate(opened)
        return opened

    async def run(self) -> None:
        delay = self.retry_min_seconds
        while True:
            manager = self.account_state.get("order_manager")
            if (manager is None or self.account_state.get("callback_bridge") is None
                    or self.account_state.get("raw_xt_trader") is None
                    or self.account_state.get("recovery_coordinator") is None):
                return
            if manager.submission_enabled:
                delay = self.retry_min_seconds
                await asyncio.sleep(self.retry_min_seconds)
                continue
            if self.blocked:
                await asyncio.sleep(self.retry_max_seconds)
                continue

            try:
                opened = await self.recover_once()
                if opened:
                    delay = self.retry_min_seconds
                else:
                    delay = min(self.retry_max_seconds, delay * 2)
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                # An unexpected supervisor error is not safe to retry forever:
                # it may indicate a programming or adapter contract failure.
                # Keep the account fail-closed until an operator resets it.
                self._set_blocked(error_text)
                gate_controller = self.account_state.get("gate_controller")
                if gate_controller is not None:
                    gate_controller.set_gate(False, reason=error_text)
                self.account_state["order_reconciliation"] = {
                    "requires_manual_review": True,
                    "startup_validation_error": error_text,
                    "recovery_blocked": True,
                    "recovery_block_reason": self.blocked_reason,
                }
                delay = self.retry_max_seconds
            await asyncio.sleep(delay)
