"""Mock-policy planning smoke check; no bundle load or gateway submission."""
from __future__ import annotations

from typing import Any, Sequence

from .account_protocol import AccountSnapshot
from .hierarchical_policy_adapter import HighDecision, HierarchicalPolicyAdapter
from .portfolio_pipeline import DecisionCycle, RebalancePlanner, StrategyCandidate


class DryRunPolicy:
    """Deterministic stand-in used only for wiring and safety checks."""

    def infer_high(self, *, candidates: Sequence[StrategyCandidate], cycle: DecisionCycle) -> HighDecision:
        return HighDecision(actions={c.symbol: ("LONG" if i < 2 else "FLAT")
                                      for i, c in enumerate(candidates)},
                            policy_version="dry-run")

    def infer_low(self, *, eligible: Sequence[StrategyCandidate], cycle: DecisionCycle,
                  high: HighDecision) -> dict[str, float]:
        return {c.symbol: .04 for c in eligible}


def run_dry_run() -> dict[str, Any]:
    account = AccountSnapshot("paper", "CNY", 100_000, 80_000, 100_000)
    cycle = DecisionCycle.create(account, cycle_id="hierarchical-dry-run", as_of="2026-11-08T01:00:00Z")
    candidates = tuple(StrategyCandidate(symbol, price, "hierarchical_rl", symbol)
                       for symbol, price in (("600000.SH", 10.0), ("000001.SZ", 20.0),
                                             ("600001.SH", 8.0)))
    adapter = HierarchicalPolicyAdapter(DryRunPolicy(), max_single_weight=.08,
                                        max_gross_weight=.80, cash_reserve=.20,
                                        bundle_hash="dry-run")
    result = adapter.infer(cycle, candidates)
    market = {"candidates": candidates}
    planned_cycle = DecisionCycle.create(account, market=market,
                                         cycle_id=cycle.cycle_id, as_of=cycle.as_of)
    intents = RebalancePlanner().plan(planned_cycle, result.target_weights,
                                      strategy_name="hierarchical_rl")
    return {"state_hash": result.state_hash, "eligible_pool": result.eligible_pool,
            "target_weights": dict(result.target_weights),
            "order_count": len(intents),
            "orders": tuple((i.symbol, i.side, i.quantity) for i in intents)}


if __name__ == "__main__":
    print(run_dry_run())
