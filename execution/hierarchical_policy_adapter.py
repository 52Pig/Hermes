"""Live adapter for a Forge two-level long-only policy.

The adapter deliberately stops at broker-neutral target weights.  It owns the
semantic boundary between the Forge policy (LONG/FLAT/EXIT plus a low-level
allocation) and Hermes' order planner.  No broker call is made here.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping, Protocol, Sequence

from .account_protocol import validate_account_state
from .portfolio_pipeline import DecisionCycle, StrategyCandidate


class HierarchicalLivePolicy(Protocol):
    def infer_high(self, *, candidates: Sequence[StrategyCandidate], cycle: DecisionCycle) -> "HighDecision": ...
    def infer_low(self, *, eligible: Sequence[StrategyCandidate], cycle: DecisionCycle,
                  high: "HighDecision") -> Mapping[str, float]: ...


@dataclass(frozen=True)
class HighDecision:
    """High-level long-only direction output.

    ``actions`` uses LONG/FLAT/EXIT.  EXIT is terminal for this cycle and is
    always forced to zero by the adapter.  ``exit_target`` is an optional
    explicit target for symbols outside the LONG pool (normally zero).
    """

    actions: Mapping[str, str]
    probabilities: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    exit_target: Mapping[str, float] = field(default_factory=dict)
    policy_version: str = ""


@dataclass(frozen=True)
class HierarchicalInferenceResult:
    high: HighDecision
    eligible_pool: tuple[str, ...]
    target_weights: Mapping[str, float]
    state_hash: str
    bundle_hash: str = ""
    policy_versions: Mapping[str, str] = field(default_factory=dict)


def _hash_state(value: Any) -> str:
    def encode(item):
        if is_dataclass(item) and not isinstance(item, type):
            return asdict(item)
        raise TypeError(f"non-canonical inference input: {type(item).__name__}")
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False, default=encode)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class HierarchicalPolicyAdapter:
    """Run high then low inference and emit post-projection target weights."""

    def __init__(self, policy: HierarchicalLivePolicy, *, max_single_weight: float = .08,
                 max_gross_weight: float = .80, cash_reserve: float = .20,
                 bundle_hash: str = "", policy_versions: Mapping[str, str] | None = None) -> None:
        if not (0 < max_single_weight <= 1 and 0 < max_gross_weight <= 1):
            raise ValueError("invalid hierarchical risk limits")
        if cash_reserve < 0 or cash_reserve >= 1:
            raise ValueError("cash_reserve must be in [0,1)")
        if max_gross_weight + cash_reserve > 1.0000001:
            raise ValueError("gross weight plus cash reserve cannot exceed one")
        self.policy = policy
        self.max_single_weight = float(max_single_weight)
        self.max_gross_weight = float(max_gross_weight)
        self.cash_reserve = float(cash_reserve)
        self.bundle_hash = bundle_hash
        self.policy_versions = dict(policy_versions or {})

    def infer(self, cycle: DecisionCycle, candidates: Sequence[StrategyCandidate]) -> HierarchicalInferenceResult:
        validate_account_state(cycle.account, cycle.positions, cycle.account.account_id)
        symbols = {c.symbol for c in candidates}
        if len(symbols) != len(candidates):
            raise ValueError("duplicate inference candidates")
        if {p.symbol.upper() for p in cycle.positions} - symbols:
            raise ValueError("candidate union must include every held symbol")
        high = self.policy.infer_high(candidates=candidates, cycle=cycle)
        actions = {str(k).upper(): str(v).upper() for k, v in high.actions.items()}
        if set(actions) != symbols:
            raise ValueError("high policy must return exactly the candidate union")
        invalid = set(actions.values()) - {"LONG", "FLAT", "EXIT"}
        if invalid:
            raise ValueError(f"invalid high-level actions: {sorted(invalid)}")
        eligible = tuple(c for c in candidates if actions.get(c.symbol, "FLAT") == "LONG")
        raw = self.policy.infer_low(eligible=eligible, cycle=cycle, high=high) if eligible else {}
        weights = {str(k).upper(): float(v) for k, v in (raw or {}).items()}
        allowed = {c.symbol for c in eligible}
        if set(weights) - allowed:
            raise ValueError("low policy returned symbol outside LONG pool")
        if any(not math.isfinite(v) or v < 0 for v in weights.values()):
            raise ValueError("low policy returned invalid weight")
        weights = {s: min(v, self.max_single_weight) for s, v in weights.items()}
        # FLAT means no new allocation. EXIT and explicit exit targets are
        # always zero/target outside the LONG pool; they can never be revived
        # by the low-level actor in this cycle.
        positions = {p.symbol.upper(): p for p in cycle.positions}
        fixed = {}
        for symbol, position in positions.items():
            action = actions.get(symbol, "FLAT")
            if action == "EXIT":
                fixed[symbol] = 0.0
            elif action != "LONG":
                current = float(position.market_value) / cycle.account.equity
                target = float(high.exit_target.get(symbol, current))
                if not math.isfinite(target) or target < 0 or target > current + 1e-8:
                    raise ValueError(f"invalid exit target for {symbol}")
                fixed[symbol] = target
        # Retained pool-external holdings consume the same account budget.
        total = sum(weights.values())
        available = max(0.0, self.max_gross_weight - sum(fixed.values()))
        if total > available and total > 0:
            weights = {s: v * available / total for s, v in weights.items()}
        weights.update(fixed)
        state = {
            "cycle_id": cycle.cycle_id, "as_of": cycle.as_of,
            "account": asdict(cycle.account), "positions": [asdict(p) for p in cycle.positions],
            "candidates": [asdict(c) for c in candidates], "market": dict(cycle.market),
            "actions": actions, "eligible_pool": [c.symbol for c in eligible],
            "weights": weights,
        }
        return HierarchicalInferenceResult(
            high=high, eligible_pool=tuple(c.symbol for c in eligible),
            target_weights=weights, state_hash=_hash_state(state),
            bundle_hash=self.bundle_hash, policy_versions=self.policy_versions,
        )


class ForgeBundlePolicy:
    """Explicit integration seam; feature_builder must reproduce training inputs.

    Forge must be installed/on PYTHONPATH. This does not invent account feature
    meanings or normalize arbitrary live vectors: those remain the audited
    feature builder's responsibility. The wrapper retains no option state;
    persistent high-level scheduling is a separate production requirement.
    """

    def __init__(self, directory, policy, feature_builder, *, expected=None):
        from inference.hierarchical_bundle import load_hierarchical_bundle
        self.manifest = load_hierarchical_bundle(directory, policy, expected=expected)
        self.policy = policy
        self.feature_builder = feature_builder
        self._pending = None

    def infer_high(self, *, candidates, cycle):
        import torch
        inputs = self.feature_builder(candidates, cycle, self.manifest)
        with torch.inference_mode():
            state = self.policy.encode(inputs["market"], inputs["holdings"])
            high = self.policy.sample_high(state, inputs["candidate_features"],
                                          inputs["valid_actions"], deterministic=True)
        labels = ("LONG", "FLAT", "EXIT")
        actions = {c.symbol: labels[int(a)] for c, a in zip(candidates, high.actions)}
        if len(actions) != len(candidates):
            raise ValueError("feature rows must align with all candidates")
        decision = HighDecision(actions, policy_version=self.manifest["policy_versions"]["high"])
        self._pending = (cycle, tuple(c.symbol for c in candidates), inputs, state, high, decision)
        return decision

    def infer_low(self, *, eligible, cycle, high):
        import torch
        if self._pending is None or self._pending[0] is not cycle or self._pending[5] is not high:
            raise ValueError("low inference requires the matching high decision")
        _, symbols, inputs, state, sampled_high, _ = self._pending
        self._pending = None
        risk = inputs["allocation_constraints"]
        with torch.inference_mode():
            low = self.policy.sample_low(state, inputs["candidate_features"], sampled_high.actions == 0,
                                        deterministic=True, **risk)
        allowed = {c.symbol for c in eligible}
        return {symbol: float(w) for symbol, w in zip(symbols, low.weights) if symbol in allowed}
