"""Read-only runtime health snapshots for monitoring and alerting."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def account_health(account_key: str, state: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a JSON-safe account health summary without querying a broker."""
    manager = state.get("order_manager")
    bridge = state.get("callback_bridge")
    supervisor = state.get("recovery_supervisor")
    if supervisor is not None and callable(getattr(supervisor, "status", None)):
        snapshot = dict(supervisor.status())
    else:
        snapshot = {
            "blocked": bool(state.get("recovery_blocked", False)),
            "blocked_reason": str(state.get("recovery_block_reason", "")),
            "connected": bool(getattr(bridge, "connected", False)),
            "subscription_ready": bool(state.get("subscription_ready", False)),
            "submission_enabled": bool(getattr(manager, "submission_enabled", False)),
            "gate_reason": getattr(manager, "submission_gate_reason", ""),
            "last_reconciliation": state.get("order_reconciliation"),
        }
    report = snapshot.get("last_reconciliation") or {}
    return {
        "account_key": str(account_key),
        "connected": bool(snapshot.get("connected", False)),
        "subscription_ready": bool(snapshot.get("subscription_ready", False)),
        "submission_enabled": bool(snapshot.get("submission_enabled", False)),
        "gate_reason": str(snapshot.get("gate_reason", "")),
        "recovery_blocked": bool(snapshot.get("blocked", False)),
        "recovery_block_reason": str(snapshot.get("blocked_reason", "")),
        "reconciliation": {
            "requires_manual_review": bool(report.get("requires_manual_review", False)),
            "unknown_broker_order_count": len(report.get("unknown_broker_orders", []) or []),
            "missing_local_order_count": len(report.get("missing_local_order_ids", []) or []),
            "gateway_error_count": len(report.get("gateway_query_errors", {}) or {}),
        },
    }


def runtime_health(accounts: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Build a deterministic aggregate snapshot suitable for a health endpoint."""
    items = [account_health(key, accounts[key]) for key in sorted(accounts)]
    return {
        "healthy": bool(items) and all(
            item["connected"]
            and item["subscription_ready"]
            and item["submission_enabled"]
            and not item["recovery_blocked"]
            and not item["reconciliation"]["requires_manual_review"]
            for item in items
        ),
        "account_count": len(items),
        "accounts": items,
    }
