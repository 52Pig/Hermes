import asyncio
import time

import pytest

from execution.account_protocol import AccountSnapshot, PositionSnapshot
from execution.order_protocol import OrderSide
from execution.portfolio_pipeline import (
    CandidateBook,
    ConcurrentDecisionPipeline,
    DecisionCycle,
    ModelScore,
    PortfolioDecisionEngine,
    PortfolioExecutionService,
    RankedCandidate,
    RebalanceConfig,
    RebalancePlanner,
    ReinforcementLearningAllocator,
    StrategyCandidate,
    fuse_scores,
)


def cycle(*, equity=100_000, positions=(), market=None):
    account = AccountSnapshot(
        account_id="ACC-1", currency="CNY", cash=equity,
        available_cash=equity, equity=equity,
    )
    return DecisionCycle.create(account, positions, market=market or {})


def candidate(symbol, price=10, strategy="s1"):
    return StrategyCandidate(symbol, price, strategy, f"sig-{symbol}-{strategy}")


def test_candidate_book_deduplicates_and_keeps_provenance():
    book = CandidateBook([candidate("000001", strategy="s1")])
    book.add_many([candidate("000001", price=11, strategy="s2")])
    item = book.freeze()[0]
    assert item.price == 11
    assert item.metadata["source_strategies"] == ("s1", "s2")
    assert item.metadata["source_signal_ids"] == ("sig-000001-s1", "sig-000001-s2")


def test_strategies_and_models_run_concurrently():
    async def strategy(name):
        await asyncio.sleep(0.05)
        return [candidate(name)]

    async def model(candidates, _cycle):
        await asyncio.sleep(0.05)
        return {item.symbol: 1.0 for item in candidates}

    started = time.perf_counter()
    pipeline = ConcurrentDecisionPipeline(
        {"a": lambda c: strategy("000001"), "b": lambda c: strategy("000002")},
        {"m1": model, "m2": model},
    )
    candidates, scores = asyncio.run(pipeline.run(cycle()))
    assert time.perf_counter() - started < 0.18
    assert {item.symbol for item in candidates} == {"000001", "000002"}
    assert {item.model_name for item in scores} == {"m1", "m2"}


def test_fuse_scores_orders_by_weighted_score():
    items = [candidate("A"), candidate("B")]
    ranked = fuse_scores(
        items,
        [ModelScore("fast", {"A": 1, "B": 0}), ModelScore("stable", {"A": 0, "B": 1})],
        {"fast": 3, "stable": 1},
    )
    assert [item.candidate.symbol for item in ranked] == ["A", "B"]
    assert ranked[0].score == pytest.approx(0.75)


def test_rl_allocator_caps_single_name_and_gross_weight():
    ranked = [RankedCandidate(candidate("A"), 1), RankedCandidate(candidate("B"), 0.5)]
    allocator = ReinforcementLearningAllocator(
        lambda _ranked, _cycle: {"A": 0.8, "B": 0.8},
        max_single_weight=0.4, max_gross_weight=0.7, cash_reserve=0.1,
    )
    weights = allocator.allocate(ranked, cycle())
    assert sum(weights.values()) == pytest.approx(0.7)
    assert max(weights.values()) <= 0.4


def test_rl_allocator_rejects_unknown_and_invalid_weights():
    ranked = [RankedCandidate(candidate("A"), 1)]
    with pytest.raises(ValueError, match="unknown"):
        ReinforcementLearningAllocator(lambda *_: {"B": 0.1}).allocate(ranked, cycle())
    with pytest.raises(ValueError, match="invalid"):
        ReinforcementLearningAllocator(lambda *_: {"A": -0.1}).allocate(ranked, cycle())
    with pytest.raises(ValueError, match="invalid"):
        ReinforcementLearningAllocator(lambda *_: {"A": float("nan")}).allocate(ranked, cycle())


def test_candidate_and_model_reject_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        candidate("A", float("inf"))
    pipeline = ConcurrentDecisionPipeline(
        {"s": lambda _cycle: [candidate("A")]},
        {"m": lambda *_: {"A": float("nan")}},
    )
    with pytest.raises(ValueError, match="non-finite"):
        asyncio.run(pipeline.run(cycle()))


def test_rebalance_uses_nav_and_limits_sell_to_available_quantity():
    position = PositionSnapshot("ACC-1", "A", 1_000, 300, 10, market_value=10_000)
    planner = RebalancePlanner(RebalanceConfig(lot_size=100, min_weight_change=0))
    orders = planner.plan(cycle(equity=100_000, positions=[position], market={"prices": {"A": 10}}), {"A": 0.0})
    assert len(orders) == 1
    assert orders[0].side == OrderSide.SELL.value
    assert orders[0].quantity == 300

    larger_nav = planner.plan(cycle(equity=200_000, market={"prices": {"A": 10}}), {"A": 0.2})
    assert larger_nav[0].quantity == 4_000


def test_missing_targets_are_not_liquidated_by_default():
    position = PositionSnapshot("ACC-1", "A", 1_000, 1_000, 10, market_value=10_000)
    planner = RebalancePlanner(RebalanceConfig(lot_size=100, min_weight_change=0))
    assert planner.plan(cycle(positions=[position], market={"prices": {"A": 10}}), {}) == ()
    planner = RebalancePlanner(RebalanceConfig(lot_size=100, min_weight_change=0, liquidate_missing_targets=True))
    assert planner.plan(cycle(positions=[position], market={"prices": {"A": 10}}), {})[0].side == OrderSide.SELL.value


def test_engine_builds_account_relative_orders():
    pipeline = ConcurrentDecisionPipeline(
        {"s": lambda _cycle: [candidate("A", 10)]},
        {"m": lambda candidates, _cycle: {candidates[0].symbol: 1}},
    )
    engine = PortfolioDecisionEngine(
        pipeline,
        ReinforcementLearningAllocator(lambda _ranked, _cycle: {"A": 0.2}),
        RebalancePlanner(RebalanceConfig(lot_size=100, min_weight_change=0)),
    )
    orders = asyncio.run(engine.build_orders(cycle(market={"prices": {"A": 10}})))
    assert orders[0].quantity == 2_000
    assert orders[0].metadata["account_equity"] == 100_000


def test_portfolio_execution_service_routes_only_through_order_manager():
    pipeline = ConcurrentDecisionPipeline(
        {"s": lambda _cycle: [candidate("A", 10)]},
        {"m": lambda candidates, _cycle: {candidates[0].symbol: 1}},
    )
    engine = PortfolioDecisionEngine(
        pipeline,
        ReinforcementLearningAllocator(lambda _ranked, _cycle: {"A": 0.1}),
        RebalancePlanner(RebalanceConfig(lot_size=100, min_weight_change=0)),
    )

    class Manager:
        def __init__(self):
            self.submitted = []

        def submit(self, intent):
            self.submitted.append(intent)
            return intent.client_order_id

    manager = Manager()
    result = asyncio.run(PortfolioExecutionService(engine, manager).execute(cycle()))
    assert len(result) == 1
    assert manager.submitted[0].quantity == 1_000
