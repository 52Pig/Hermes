from execution.health import account_health, runtime_health


class Manager:
    submission_enabled = True
    submission_gate_reason = ""


class Bridge:
    connected = True


def test_runtime_health_is_safe_and_aggregated():
    state = {
        "order_manager": Manager(),
        "callback_bridge": Bridge(),
        "subscription_ready": True,
        "order_reconciliation": {
            "requires_manual_review": False,
            "unknown_broker_orders": [],
            "missing_local_order_ids": [],
        },
    }
    item = account_health("acc-1", state)
    assert item["account_key"] == "acc-1"
    assert item["submission_enabled"]
    assert item["reconciliation"]["gateway_error_count"] == 0

    health = runtime_health({"z": state, "a": state})
    assert health["healthy"]
    assert health["account_count"] == 2
    assert [row["account_key"] for row in health["accounts"]] == ["a", "z"]


def test_runtime_health_marks_manual_review_unhealthy():
    state = {
        "recovery_blocked": True,
        "recovery_block_reason": "unknown broker order",
        "order_reconciliation": {
            "requires_manual_review": True,
            "unknown_broker_orders": [{"id": "B-1"}],
            "missing_local_order_ids": ["L-1"],
            "gateway_query_errors": {"ctp": "offline"},
        },
    }
    health = runtime_health({"acc": state})
    assert not health["healthy"]
    row = health["accounts"][0]
    assert row["recovery_blocked"]
    assert row["reconciliation"] == {
        "requires_manual_review": True,
        "unknown_broker_order_count": 1,
        "missing_local_order_count": 1,
        "gateway_error_count": 1,
    }
