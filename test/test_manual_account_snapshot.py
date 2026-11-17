import json
from datetime import datetime, timezone

import pytest

from execution.manual_account_snapshot import (
    load_manual_account_snapshot,
    write_manual_snapshot_template,
)


NOW = datetime(2026, 11, 8, 7, 0, tzinfo=timezone.utc)


def _write(path, **updates):
    payload = {
        "account_id": "A-100000",
        "currency": "CNY",
        "as_of": "2026-11-08T14:50:00+08:00",
        "cash": 100000.0,
        "available_cash": 90000.0,
        "equity": 110000.0,
        "frozen_cash": 1000.0,
        "margin_used": 0.0,
        "positions": [{
            "symbol": "000001.SZ",
            "quantity": 100,
            "available_quantity": 100,
            "average_price": 10.0,
            "market_value": 2000.0,
            "today_quantity": 0,
            "yesterday_quantity": 100,
        }],
    }
    payload.update(updates)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_valid_snapshot_and_normalize_timestamp(tmp_path):
    path = tmp_path / "account.json"
    _write(path)

    account = load_manual_account_snapshot(path, expected_account_id="A-100000", now=NOW)

    assert account.account_id == "A-100000"
    assert account.currency == "CNY"
    assert account.available_cash == pytest.approx(90000)
    assert account.positions[0].symbol == "000001.SZ"
    assert account.as_of == "2026-11-08T06:50:00+00:00"
    assert account.metadata["source"] == "manual_broker_export"


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"account_id": "OTHER"}, "account_snapshot_mismatch"),
        ({"as_of": "2026-11-07T14:50:00+08:00"}, "account_snapshot_stale"),
        ({"as_of": "2026-11-08T15:02:00+08:00"}, "account_snapshot_from_future"),
        ({"available_cash": 100001.0}, "available_cash_exceeds_cash"),
        ({"positions": [{
            "symbol": "000001.SZ", "quantity": 100, "available_quantity": 101,
            "average_price": 10.0,
        }]}, "available_exceeds_position:000001.SZ"),
    ],
)
def test_rejects_unsafe_snapshot_values(tmp_path, updates, error):
    path = tmp_path / "account.json"
    _write(path, **updates)

    with pytest.raises(ValueError, match=error):
        load_manual_account_snapshot(path, expected_account_id="A-100000", now=NOW)


def test_rejects_duplicate_positions(tmp_path):
    path = tmp_path / "account.json"
    duplicate = [
        {"symbol": "000001.SZ", "quantity": 100, "available_quantity": 100, "average_price": 10},
        {"symbol": "000001.SZ", "quantity": 50, "available_quantity": 50, "average_price": 11},
    ]
    _write(path, positions=duplicate)

    with pytest.raises(ValueError, match="duplicate_position_snapshot:000001.SZ"):
        load_manual_account_snapshot(path, expected_account_id="A-100000", now=NOW)


def test_template_is_readable_and_has_empty_positions(tmp_path):
    path = tmp_path / "template.json"
    write_manual_snapshot_template(path, "A-100000")

    account = load_manual_account_snapshot(path, expected_account_id="A-100000", now=datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc))

    assert account.positions == ()
    assert account.equity == pytest.approx(100000)
