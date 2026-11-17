import asyncio
import sqlite3

from execution.account_protocol import AccountSnapshot
from execution.portfolio_pipeline import (
    ConcurrentDecisionPipeline, DecisionCycle, PortfolioDecisionEngine,
    RebalanceConfig, RebalancePlanner, ReinforcementLearningAllocator,
    StrategyCandidate,
)
from execution.portfolio_shadow import PortfolioShadowRunner, shadow_summary


def _cycle():
    account = AccountSnapshot("ACC-1", "CNY", 100_000, 100_000, 100_000)
    return DecisionCycle.create(
        account, cycle_id="shadow-1", as_of="2026-11-08T09:25:00+08:00",
        market={"prices": {"A": 10}},
    )


def test_shadow_builds_orders_without_order_manager(tmp_path):
    pipeline = ConcurrentDecisionPipeline(
        {"s": lambda _cycle: [StrategyCandidate("A", 10, "s", "sig-A")]},
        {"m": lambda candidates, _cycle: {candidates[0].symbol: 1}},
    )
    engine = PortfolioDecisionEngine(
        pipeline,
        ReinforcementLearningAllocator(lambda _ranked, _cycle: {"A": 0.1}),
        RebalancePlanner(RebalanceConfig(min_weight_change=0)),
    )
    runner = PortfolioShadowRunner(engine, audit_path=tmp_path / "shadow.sqlite3")
    result = asyncio.run(runner.evaluate(_cycle()))
    assert result.order_count == 1
    assert result.input_hash
    summary = shadow_summary(result)
    assert summary["orders"][0]["symbol"] == "A"
    assert summary["status"] == "PASS"
    assert summary["repeated"] is False
    runner.close()


def test_shadow_marks_same_cycle_repeat_and_rejects_changed_snapshot(tmp_path):
    pipeline = ConcurrentDecisionPipeline(
        {"s": lambda _cycle: [StrategyCandidate("A", 10, "s", "sig-A")]},
        {"m": lambda candidates, _cycle: {candidates[0].symbol: 1}},
    )
    engine = PortfolioDecisionEngine(
        pipeline,
        ReinforcementLearningAllocator(lambda _ranked, _cycle: {"A": 0.1}),
        RebalancePlanner(RebalanceConfig(min_weight_change=0)),
    )
    runner = PortfolioShadowRunner(engine, audit_path=tmp_path / "shadow.sqlite3")
    original = _cycle()
    first = asyncio.run(runner.evaluate(original))
    repeated = asyncio.run(runner.evaluate(original))
    changed = original
    changed = DecisionCycle.create(
        changed.account, cycle_id=changed.cycle_id, as_of=changed.as_of,
        market={"prices": {"A": 11}},
    )
    rejected = asyncio.run(runner.evaluate(changed))

    assert first.input_hash == repeated.input_hash
    assert repeated.repeated is True
    assert rejected.status == "FAIL"
    assert rejected.failure_code == "cycle_input_changed"
    assert rejected.order_count == 0
    runner.close()


def test_shadow_restores_cached_decision_and_rejects_changed_input_after_restart(tmp_path):
    calls = []

    def strategy(_cycle):
        calls.append("strategy")
        return [StrategyCandidate("A", 10, "s", "sig-A")]

    pipeline = ConcurrentDecisionPipeline(
        {"s": strategy},
        {"m": lambda candidates, _cycle: {candidates[0].symbol: 1}},
    )
    engine = PortfolioDecisionEngine(
        pipeline,
        ReinforcementLearningAllocator(lambda _ranked, _cycle: {"A": 0.1}),
        RebalancePlanner(RebalanceConfig(min_weight_change=0)),
    )
    database = tmp_path / "shadow.sqlite3"
    original = _cycle()

    first_runner = PortfolioShadowRunner(engine, audit_path=database)
    first = asyncio.run(first_runner.evaluate(original))
    first_runner.close()

    restarted_runner = PortfolioShadowRunner(engine, audit_path=database)
    repeated = asyncio.run(restarted_runner.evaluate(original))
    changed = DecisionCycle.create(
        original.account, cycle_id=original.cycle_id, as_of=original.as_of,
        market={"prices": {"A": 11}},
    )
    rejected = asyncio.run(restarted_runner.evaluate(changed))

    assert repeated.repeated is True
    assert repeated.orders == first.orders
    assert len(calls) == 1  # restart did not recompute a completed cycle
    assert rejected.status == "FAIL"
    assert rejected.failure_code == "cycle_input_changed"
    restarted_runner.close()

    with sqlite3.connect(database) as connection:
        event_types = [row[0] for row in connection.execute(
            "SELECT event_type FROM shadow_events WHERE cycle_id = ? ORDER BY event_id",
            (original.cycle_id,),
        )]
    assert event_types == ["evaluation_passed", "cycle_repeated", "cycle_input_changed"]


def test_shadow_records_engine_failure_as_structured_result(tmp_path):
    pipeline = ConcurrentDecisionPipeline(
        {"s": lambda _cycle: [StrategyCandidate("A", 10, "s", "sig-A")]},
        {},
    )
    engine = PortfolioDecisionEngine(
        pipeline,
        ReinforcementLearningAllocator(lambda _ranked, _cycle: {"UNKNOWN": 0.1}),
        RebalancePlanner(),
    )
    runner = PortfolioShadowRunner(engine, audit_path=tmp_path / "shadow.sqlite3")
    result = asyncio.run(runner.evaluate(_cycle()))

    assert result.status == "FAIL"
    assert result.failure_code == "ValueError"
    assert "unknown symbols" in result.failure_message
    assert result.orders == ()
    assert shadow_summary(result)["status"] == "FAIL"
    runner.close()
