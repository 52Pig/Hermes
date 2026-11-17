"""Concurrent strategy/model pipeline and account-relative portfolio planning.

The pipeline deliberately stops at broker-neutral ``OrderIntent`` objects.
Strategies and models may run concurrently, while account state, target weights,
and rebalance quantities are resolved once per decision cycle.  An injected
reinforcement-learning policy can provide weights; deterministic constraints
still validate and cap its output before anything reaches OrderManager.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, Mapping, Optional, Protocol, Sequence

from .account_protocol import AccountSnapshot, PositionSnapshot
from .order_protocol import OrderIntent, OrderSide, OrderType


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class DecisionCycle:
    """Immutable inputs shared by every strategy/model in one decision cycle."""

    cycle_id: str
    as_of: str
    account: AccountSnapshot
    positions: tuple[PositionSnapshot, ...] = ()
    market: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        account: AccountSnapshot,
        positions: Iterable[PositionSnapshot] = (),
        *,
        market: Optional[Mapping[str, Any]] = None,
        cycle_id: Optional[str] = None,
        as_of: Optional[str] = None,
    ) -> "DecisionCycle":
        if account.equity <= 0:
            raise ValueError("decision cycle requires positive account equity")
        return cls(
            cycle_id=str(cycle_id or uuid.uuid4().hex),
            as_of=str(as_of or _utc_now()),
            account=account,
            positions=tuple(positions),
            market=dict(market or {}),
        )


@dataclass(frozen=True)
class StrategyCandidate:
    symbol: str
    price: float
    strategy_name: str
    signal_id: str
    features: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip().upper()
        if not symbol:
            raise ValueError("candidate symbol is required")
        if not math.isfinite(float(self.price)) or float(self.price) <= 0:
            raise ValueError("candidate price must be finite and positive")
        if not str(self.strategy_name).strip():
            raise ValueError("candidate strategy_name is required")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "price", float(self.price))
        object.__setattr__(self, "strategy_name", str(self.strategy_name).strip())
        object.__setattr__(self, "signal_id", str(self.signal_id or ""))


class RankingModel(Protocol):
    def score(self, candidates: Sequence[StrategyCandidate], cycle: DecisionCycle) -> Mapping[str, float]:
        ...


@dataclass(frozen=True)
class ModelScore:
    model_name: str
    scores: Mapping[str, float]


@dataclass(frozen=True)
class RankedCandidate:
    candidate: StrategyCandidate
    score: float
    model_scores: Mapping[str, float] = field(default_factory=dict)


class CandidateBook:
    """Merge concurrent strategy output by symbol for one frozen cycle."""

    def __init__(self, candidates: Iterable[StrategyCandidate] = ()) -> None:
        self._items: Dict[str, StrategyCandidate] = {}
        self.add_many(candidates)

    def add_many(self, candidates: Iterable[StrategyCandidate]) -> None:
        for candidate in candidates:
            previous = self._items.get(candidate.symbol)
            if previous is None:
                self._items[candidate.symbol] = candidate
                continue
            # Preserve all strategy provenance while keeping one price/features
            # record for deterministic downstream model input.
            metadata = dict(previous.metadata)
            sources = list(metadata.get("source_strategies", [previous.strategy_name]))
            if candidate.strategy_name not in sources:
                sources.append(candidate.strategy_name)
            metadata["source_strategies"] = tuple(sources)
            metadata["source_signal_ids"] = tuple(
                list(metadata.get("source_signal_ids", [previous.signal_id]))
                + ([candidate.signal_id] if candidate.signal_id else [])
            )
            features = dict(previous.features)
            features.update(candidate.features)
            self._items[candidate.symbol] = StrategyCandidate(
                symbol=previous.symbol,
                price=candidate.price,
                strategy_name=previous.strategy_name,
                signal_id=previous.signal_id,
                features=features,
                metadata=metadata,
            )

    def freeze(self) -> tuple[StrategyCandidate, ...]:
        return tuple(self._items[key] for key in sorted(self._items))


async def _call(value: Any, *args: Any) -> Any:
    method = value
    if hasattr(value, "generate"):
        method = value.generate
    elif hasattr(value, "score"):
        method = value.score
    if inspect.iscoroutinefunction(method):
        return await method(*args)
    result = await asyncio.to_thread(method, *args)
    if inspect.isawaitable(result):
        return await result
    return result


class ConcurrentDecisionPipeline:
    """Run strategy generation and model scoring concurrently per cycle."""

    def __init__(
        self,
        strategies: Mapping[str, Callable[..., Any]],
        models: Mapping[str, RankingModel | Callable[..., Any]],
    ) -> None:
        self.strategies = dict(strategies)
        self.models = dict(models)

    async def collect_candidates(self, cycle: DecisionCycle) -> tuple[StrategyCandidate, ...]:
        tasks = [asyncio.create_task(_call(strategy, cycle), name=f"strategy:{name}")
                 for name, strategy in self.strategies.items()]
        results = await asyncio.gather(*tasks)
        book = CandidateBook()
        for result in results:
            book.add_many(result or ())
        return book.freeze()

    async def score_candidates(
        self, candidates: Sequence[StrategyCandidate], cycle: DecisionCycle
    ) -> tuple[ModelScore, ...]:
        tasks = [asyncio.create_task(_call(model, candidates, cycle), name=f"model:{name}")
                 for name, model in self.models.items()]
        results = await asyncio.gather(*tasks)
        scores = []
        for name, result in zip(self.models, results):
            normalised = {str(key).upper(): float(value) for key, value in (result or {}).items()}
            if any(not math.isfinite(value) for value in normalised.values()):
                raise ValueError(f"model {name} returned a non-finite score")
            scores.append(ModelScore(name, normalised))
        return tuple(scores)

    async def run(self, cycle: DecisionCycle) -> tuple[tuple[StrategyCandidate, ...], tuple[ModelScore, ...]]:
        candidates = await self.collect_candidates(cycle)
        return candidates, await self.score_candidates(candidates, cycle)


def fuse_scores(
    candidates: Sequence[StrategyCandidate], model_scores: Sequence[ModelScore],
    weights: Optional[Mapping[str, float]] = None,
) -> tuple[RankedCandidate, ...]:
    """Weighted average model scores; missing scores count as zero."""
    if not model_scores:
        return tuple(RankedCandidate(c, 0.0) for c in candidates)
    model_weights = {score.model_name: float((weights or {}).get(score.model_name, 1.0))
                     for score in model_scores}
    if (any(not math.isfinite(value) or value < 0 for value in model_weights.values())
            or sum(model_weights.values()) <= 0):
        raise ValueError("model weights must be non-negative with a positive total")
    total = sum(model_weights.values())
    result = []
    for candidate in candidates:
        per_model = {score.model_name: float(score.scores.get(candidate.symbol, 0.0))
                     for score in model_scores}
        combined = sum(per_model[name] * model_weights[name] for name in per_model) / total
        result.append(RankedCandidate(candidate, combined, per_model))
    return tuple(sorted(result, key=lambda item: (-item.score, item.candidate.symbol)))


class WeightPolicy(Protocol):
    def allocate(self, ranked: Sequence[RankedCandidate], cycle: DecisionCycle) -> Mapping[str, float]:
        ...


class ReinforcementLearningAllocator:
    """Adapter for an offline-trained RL policy with hard live constraints."""

    def __init__(self, policy: WeightPolicy | Callable[..., Mapping[str, float]], *,
                 max_single_weight: float = 0.20, max_gross_weight: float = 1.0,
                 cash_reserve: float = 0.05) -> None:
        self.policy = policy
        self.max_single_weight = float(max_single_weight)
        self.max_gross_weight = float(max_gross_weight)
        self.cash_reserve = float(cash_reserve)
        if not 0 <= self.cash_reserve < 1:
            raise ValueError("cash_reserve must be in [0, 1)")
        if self.max_single_weight <= 0 or self.max_gross_weight <= 0:
            raise ValueError("weight limits must be positive")

    def allocate(self, ranked: Sequence[RankedCandidate], cycle: DecisionCycle) -> Dict[str, float]:
        raw = self.policy.allocate(ranked, cycle) if hasattr(self.policy, "allocate") else self.policy(ranked, cycle)
        weights = {str(symbol).upper(): float(value) for symbol, value in (raw or {}).items()}
        allowed = {item.candidate.symbol for item in ranked}
        unknown = set(weights) - allowed
        if unknown:
            raise ValueError(f"RL policy returned unknown symbols: {sorted(unknown)}")
        if any(not math.isfinite(value) or value < 0 for value in weights.values()):
            raise ValueError("RL policy returned invalid target weight")
        weights = {symbol: min(value, self.max_single_weight) for symbol, value in weights.items()}
        cap = min(self.max_gross_weight, 1.0 - self.cash_reserve)
        total = sum(weights.values())
        if total > cap and total > 0:
            scale = cap / total
            weights = {symbol: value * scale for symbol, value in weights.items()}
        return weights


@dataclass(frozen=True)
class RebalanceConfig:
    lot_size: int = 100
    min_weight_change: float = 0.005
    order_type: str = OrderType.LIMIT.value
    execution_channel: str = ""
    liquidate_missing_targets: bool = False


class RebalancePlanner:
    """Convert account-relative target weights into broker-neutral intents."""

    def __init__(self, config: Optional[RebalanceConfig] = None) -> None:
        self.config = config or RebalanceConfig()
        if self.config.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if self.config.min_weight_change < 0:
            raise ValueError("min_weight_change cannot be negative")

    def plan(self, cycle: DecisionCycle, weights: Mapping[str, float], *,
             strategy_name: str = "portfolio_allocator") -> tuple[OrderIntent, ...]:
        if cycle.account.equity <= 0:
            raise ValueError("account equity must be positive")
        prices = {}
        for item in cycle.market.get("candidates", ()):
            if isinstance(item, StrategyCandidate):
                prices[item.symbol] = item.price
            elif isinstance(item, Mapping):
                prices[str(item["symbol"]).upper()] = float(item["price"])
        # The cycle's candidate list is normally passed in market metadata by
        # the coordinator; callers may also supply prices directly.
        for symbol, value in cycle.market.get("prices", {}).items():
            price = float(value)
            if not math.isfinite(price) or price <= 0:
                raise ValueError(f"invalid price for {str(symbol).upper()}")
            prices[str(symbol).upper()] = price
        positions = {position.symbol.upper(): position for position in cycle.positions}
        intents = []
        all_symbols = {str(symbol).upper() for symbol in weights}
        if self.config.liquidate_missing_targets:
            all_symbols |= set(positions)
        for symbol in sorted(all_symbols):
            target = float(weights.get(symbol, 0.0))
            symbol = str(symbol).upper()
            if not math.isfinite(target) or target < 0:
                raise ValueError(f"invalid target weight for {symbol}")
            price = float(prices.get(symbol, 0))
            if price <= 0:
                raise ValueError(f"missing price for target symbol {symbol}")
            current = positions.get(symbol)
            current_value = float(current.market_value) if current else 0.0
            current_weight = current_value / cycle.account.equity
            delta = float(target) - current_weight
            if abs(delta) < self.config.min_weight_change:
                continue
            raw_quantity = abs(delta) * cycle.account.equity / price
            quantity = int(raw_quantity // self.config.lot_size) * self.config.lot_size
            if quantity <= 0:
                continue
            side = OrderSide.BUY.value if delta > 0 else OrderSide.SELL.value
            if side == OrderSide.SELL.value and current is not None:
                quantity = min(quantity, int(current.available_quantity) // self.config.lot_size * self.config.lot_size)
            if quantity <= 0:
                continue
            metadata = {
                "target_weight": float(target),
                "current_weight": current_weight,
                "reference_price": price,
                "cycle_id": cycle.cycle_id,
                "account_equity": float(cycle.account.equity),
                "execution_channel": self.config.execution_channel,
            }
            intents.append(OrderIntent(
                account_id=cycle.account.account_id,
                strategy_name=strategy_name,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=self.config.order_type,
                limit_price=price if self.config.order_type == OrderType.LIMIT.value else None,
                signal_id=f"{cycle.cycle_id}:{symbol}",
                signal_timestamp=cycle.as_of,
                metadata=metadata,
            ))
        return tuple(intents)


class PortfolioDecisionEngine:
    """Compose pipeline, score fusion, RL allocation, and rebalance planning."""

    def __init__(self, pipeline: ConcurrentDecisionPipeline,
                 allocator: ReinforcementLearningAllocator,
                 planner: Optional[RebalancePlanner] = None,
                 model_weights: Optional[Mapping[str, float]] = None) -> None:
        self.pipeline = pipeline
        self.allocator = allocator
        self.planner = planner or RebalancePlanner()
        self.model_weights = dict(model_weights or {})

    async def build_orders(self, cycle: DecisionCycle) -> tuple[OrderIntent, ...]:
        candidates, model_scores = await self.pipeline.run(cycle)
        ranked = fuse_scores(candidates, model_scores, self.model_weights)
        weights = self.allocator.allocate(ranked, cycle)
        market = dict(cycle.market)
        market["candidates"] = candidates
        priced_cycle = DecisionCycle(
            cycle_id=cycle.cycle_id,
            as_of=cycle.as_of,
            account=cycle.account,
            positions=cycle.positions,
            market=market,
        )
        return self.planner.plan(priced_cycle, weights)


class PortfolioExecutionService:
    """Submit portfolio intents only through the durable OrderManager boundary."""

    def __init__(self, engine: PortfolioDecisionEngine, order_manager: Any) -> None:
        if not callable(getattr(order_manager, "submit", None)):
            raise TypeError("order_manager must provide submit(intent)")
        self.engine = engine
        self.order_manager = order_manager
        self._cycle_locks: Dict[str, asyncio.Lock] = {}

    async def execute(self, cycle: DecisionCycle) -> tuple[Any, ...]:
        # Calls for the same cycle serialize. Deterministic order IDs in
        # OrderManager make a repeated cycle idempotent across restarts too.
        lock = self._cycle_locks.setdefault(cycle.cycle_id, asyncio.Lock())
        async with lock:
            intents = await self.engine.build_orders(cycle)
            return tuple(self.order_manager.submit(intent) for intent in intents)
