from execution.account_protocol import AccountSnapshot, PositionSnapshot
from execution.portfolio_pipeline import DecisionCycle, StrategyCandidate
from execution.hierarchical_policy_adapter import HighDecision, HierarchicalPolicyAdapter


class FakePolicy:
    def infer_high(self, *, candidates, cycle):
        return HighDecision(actions={"AAA": "LONG", "BBB": "EXIT", "CCC": "FLAT"},
                            policy_version="h1")

    def infer_low(self, *, eligible, cycle, high):
        assert [c.symbol for c in eligible] == ["AAA"]
        return {"AAA": .50}


def _cycle():
    account = AccountSnapshot("a", "CNY", 100000, 90000, 100000)
    positions = (PositionSnapshot("a", "BBB", 1000, 0, 10, 10000),)
    return DecisionCycle.create(account, positions, cycle_id="c1", as_of="2026-11-08T01:00:00Z")


def test_high_pool_is_enforced_and_exit_cannot_be_reintroduced():
    candidates = [StrategyCandidate("AAA", 10, "x", "1"),
                  StrategyCandidate("BBB", 10, "x", "2"),
                  StrategyCandidate("CCC", 10, "x", "3")]
    result = HierarchicalPolicyAdapter(FakePolicy()).infer(_cycle(), candidates)
    assert result.eligible_pool == ("AAA",)
    assert result.target_weights["AAA"] == .08
    assert result.target_weights["BBB"] == 0
    assert "CCC" not in result.target_weights


def test_low_policy_unknown_symbol_fails():
    class Bad(FakePolicy):
        def infer_high(self, *, candidates, cycle):
            return HighDecision(actions={c.symbol: ("LONG" if c.symbol == "AAA" else "FLAT") for c in candidates})
        def infer_low(self, *, eligible, cycle, high):
            return {"ZZZ": .1}
    candidates = [StrategyCandidate("AAA", 10, "x", "1"), StrategyCandidate("BBB", 10, "x", "2")]
    try:
        HierarchicalPolicyAdapter(Bad()).infer(_cycle(), candidates)
    except ValueError as exc:
        assert "outside LONG pool" in str(exc)
    else:
        raise AssertionError("expected unknown low symbol rejection")
