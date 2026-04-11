from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from deferralx.schema import Record
from deferralx.utility import (
    UtilityConfig,
    expected_utility_if_accept,
    expected_utility_if_escalate,
    realized_utility,
)


@dataclass
class Decision:
    accept: bool
    p_correct: float


class BasePolicy:
    name = "base"

    def fit(self, records: list[Record], config: UtilityConfig) -> None:
        _ = records
        _ = config

    def decide(self, record: Record, config: UtilityConfig) -> Decision:
        raise NotImplementedError


class AlwaysEscalatePolicy(BasePolicy):
    name = "always_escalate"

    def decide(self, record: Record, config: UtilityConfig) -> Decision:
        _ = config
        return Decision(accept=False, p_correct=record.calibrated_confidence)


class GlobalThresholdPolicy(BasePolicy):
    name = "global_threshold"

    def __init__(self, step: float = 0.01) -> None:
        self.step = step
        self.threshold = 1.0

    def fit(self, records: list[Record], config: UtilityConfig) -> None:
        best_t = 1.0
        best_u = -10e9
        for t in _threshold_grid(self.step):
            total = 0.0
            for r in records:
                accept = r.calibrated_confidence >= t
                total += realized_utility(r, accept, config)
            avg = total / max(1, len(records))
            if avg > best_u:
                best_u = avg
                best_t = t
        self.threshold = best_t

    def decide(self, record: Record, config: UtilityConfig) -> Decision:
        _ = config
        return Decision(
            accept=record.calibrated_confidence >= self.threshold,
            p_correct=record.calibrated_confidence,
        )


class GroupThresholdPolicy(BasePolicy):
    def __init__(self, mode: str, step: float = 0.01) -> None:
        if mode not in {"domain", "profile", "domain_profile"}:
            raise ValueError("mode must be one of: domain, profile, domain_profile")
        self.mode = mode
        self.step = step
        self.thresholds: dict[str, float] = {}
        self.default_threshold = 1.0
        self.name = f"{mode}_threshold"

    def _group_key(self, record: Record) -> str:
        if self.mode == "domain":
            return record.domain
        if self.mode == "profile":
            return record.user_profile
        return f"{record.domain}::{record.user_profile}"

    def fit(self, records: list[Record], config: UtilityConfig) -> None:
        grouped: dict[str, list[Record]] = defaultdict(list)
        for r in records:
            grouped[self._group_key(r)].append(r)

        self.thresholds = {}
        for key, group in grouped.items():
            best_t = 1.0
            best_u = -10e9
            for t in _threshold_grid(self.step):
                total = 0.0
                for r in group:
                    accept = r.calibrated_confidence >= t
                    total += realized_utility(r, accept, config)
                avg = total / max(1, len(group))
                if avg > best_u:
                    best_u = avg
                    best_t = t
            self.thresholds[key] = best_t

        if records:
            global_policy = GlobalThresholdPolicy(step=self.step)
            global_policy.fit(records, config)
            self.default_threshold = global_policy.threshold

    def decide(self, record: Record, config: UtilityConfig) -> Decision:
        _ = config
        threshold = self.thresholds.get(self._group_key(record), self.default_threshold)
        return Decision(
            accept=record.calibrated_confidence >= threshold,
            p_correct=record.calibrated_confidence,
        )


class LinearUtilityRouterPolicy(BasePolicy):
    name = "learned_utility_router"

    def __init__(
        self,
        learning_rate: float = 0.10,
        epochs: int = 450,
        l2: float = 0.001,
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.domains: list[str] = []
        self.profiles: list[str] = []
        self.weights: list[float] = []

    def fit(self, records: list[Record], config: UtilityConfig) -> None:
        _ = config
        if not records:
            return

        self.domains = sorted({r.domain for r in records})
        self.profiles = sorted({r.user_profile for r in records})

        xs = [self._features(r) for r in records]
        ys = [float(r.correctness) for r in records]

        d = len(xs[0])
        self.weights = [0.0] * d

        for _ in range(self.epochs):
            grad = [0.0] * d
            for x, y in zip(xs, ys):
                p = _sigmoid(dot(self.weights, x))
                err = p - y
                for i in range(d):
                    grad[i] += err * x[i]

            n = float(len(xs))
            for i in range(d):
                grad_i = grad[i] / n + self.l2 * self.weights[i]
                self.weights[i] -= self.learning_rate * grad_i

    def _features(self, record: Record) -> list[float]:
        x = [
            1.0,
            record.calibrated_confidence,
            record.p_internal,
            record.p_verbal,
            record.agreement,
            record.speed_fast,
        ]

        for d in self.domains:
            x.append(1.0 if record.domain == d else 0.0)
        for p in self.profiles:
            x.append(1.0 if record.user_profile == p else 0.0)
        return x

    def predict_correctness(self, record: Record) -> float:
        if not self.weights:
            return record.calibrated_confidence
        return _sigmoid(dot(self.weights, self._features(record)))

    def decide(self, record: Record, config: UtilityConfig) -> Decision:
        p_correct = self.predict_correctness(record)
        eu_accept = expected_utility_if_accept(record, p_correct, config)
        eu_escalate = expected_utility_if_escalate(record, config)
        return Decision(accept=eu_accept >= eu_escalate, p_correct=p_correct)


class RTRStyleThresholdRouterPolicy(BasePolicy):
    """
    Route-To-Reason-inspired baseline adapted to human deferral:
    1) learn a performance predictor p(correct | x)
    2) learn a single threshold on predicted performance for routing.
    """

    name = "rtr_style_threshold_router"

    def __init__(self, step: float = 0.01) -> None:
        self.step = step
        self.threshold = 1.0
        self.predictor = LinearUtilityRouterPolicy(
            learning_rate=0.12,
            epochs=500,
            l2=0.001,
        )

    def fit(self, records: list[Record], config: UtilityConfig) -> None:
        self.predictor.fit(records, config)
        best_t = 1.0
        best_u = -10e9
        for t in _threshold_grid(self.step):
            total = 0.0
            for r in records:
                p = self.predictor.predict_correctness(r)
                total += realized_utility(r, p >= t, config)
            avg = total / max(1, len(records))
            if avg > best_u:
                best_u = avg
                best_t = t
        self.threshold = best_t

    def decide(self, record: Record, config: UtilityConfig) -> Decision:
        _ = config
        p = self.predictor.predict_correctness(record)
        return Decision(accept=p >= self.threshold, p_correct=p)


def _threshold_grid(step: float) -> list[float]:
    values: list[float] = []
    x = 0.0
    while x <= 1.000001:
        values.append(round(x, 6))
        x += step
    if values[-1] != 1.0:
        values.append(1.0)
    return values


def dot(a: list[float], b: list[float]) -> float:
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def _sigmoid(value: float) -> float:
    # Numerically stable clamp for pure-Python math.exp.
    if value < -35.0:
        return 0.0
    if value > 35.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))
