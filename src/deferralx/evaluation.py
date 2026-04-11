from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from deferralx.policies import BasePolicy, Decision
from deferralx.schema import Record
from deferralx.utility import UtilityConfig, realized_utility


@dataclass
class DecisionRow:
    policy: str
    example_id: str
    domain: str
    user_profile: str
    correctness: int
    severe_if_wrong: int
    accept: int
    p_correct: float
    utility: float


@dataclass
class MetricsRow:
    policy: str
    slice_name: str
    n: int
    utility_mean: float
    coverage: float
    escalation_rate: float
    accepted_accuracy: float | None
    severe_error_rate: float


def run_policy(
    policy: BasePolicy,
    records: list[Record],
    config: UtilityConfig,
) -> list[DecisionRow]:
    rows: list[DecisionRow] = []
    for r in records:
        d: Decision = policy.decide(r, config)
        utility = realized_utility(r, d.accept, config)
        rows.append(
            DecisionRow(
                policy=policy.name,
                example_id=r.example_id,
                domain=r.domain,
                user_profile=r.user_profile,
                correctness=r.correctness,
                severe_if_wrong=r.severe_if_wrong,
                accept=1 if d.accept else 0,
                p_correct=d.p_correct,
                utility=utility,
            )
        )
    return rows


def compute_metrics(policy_name: str, slice_name: str, rows: list[DecisionRow]) -> MetricsRow:
    n = len(rows)
    if n == 0:
        return MetricsRow(
            policy=policy_name,
            slice_name=slice_name,
            n=0,
            utility_mean=0.0,
            coverage=0.0,
            escalation_rate=0.0,
            accepted_accuracy=None,
            severe_error_rate=0.0,
        )

    accepted = [r for r in rows if r.accept == 1]
    accepted_n = len(accepted)
    accepted_correct = sum(r.correctness for r in accepted)
    severe_wrong_accepted = sum(
        1 for r in accepted if (r.correctness == 0 and r.severe_if_wrong == 1)
    )

    utility_mean = sum(r.utility for r in rows) / n
    coverage = accepted_n / n
    escalation_rate = 1.0 - coverage
    severe_error_rate = severe_wrong_accepted / n

    accepted_accuracy: float | None = None
    if accepted_n > 0:
        accepted_accuracy = accepted_correct / accepted_n

    return MetricsRow(
        policy=policy_name,
        slice_name=slice_name,
        n=n,
        utility_mean=utility_mean,
        coverage=coverage,
        escalation_rate=escalation_rate,
        accepted_accuracy=accepted_accuracy,
        severe_error_rate=severe_error_rate,
    )


def aggregate_by_slice(policy_name: str, rows: list[DecisionRow]) -> dict[str, list[MetricsRow]]:
    out: dict[str, list[MetricsRow]] = {}

    out["overall"] = [compute_metrics(policy_name, "overall", rows)]

    by_domain = _group(rows, key_fn=lambda r: r.domain)
    out["domain"] = [compute_metrics(policy_name, k, v) for k, v in sorted(by_domain.items())]

    by_profile = _group(rows, key_fn=lambda r: r.user_profile)
    out["profile"] = [
        compute_metrics(policy_name, k, v) for k, v in sorted(by_profile.items())
    ]

    by_domain_profile = _group(rows, key_fn=lambda r: f"{r.domain}::{r.user_profile}")
    out["domain_profile"] = [
        compute_metrics(policy_name, k, v) for k, v in sorted(by_domain_profile.items())
    ]

    return out


def bootstrap_ci(
    rows: list[DecisionRow],
    n_boot: int = 300,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    if len(rows) == 0:
        return {
            "utility_mean": (0.0, 0.0),
            "coverage": (0.0, 0.0),
            "severe_error_rate": (0.0, 0.0),
        }

    rng = random.Random(seed)
    util_samples: list[float] = []
    cov_samples: list[float] = []
    severe_samples: list[float] = []

    n = len(rows)
    for _ in range(n_boot):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        m = compute_metrics(rows[0].policy, "bootstrap", sample)
        util_samples.append(m.utility_mean)
        cov_samples.append(m.coverage)
        severe_samples.append(m.severe_error_rate)

    return {
        "utility_mean": _percentile_interval(util_samples, 0.025, 0.975),
        "coverage": _percentile_interval(cov_samples, 0.025, 0.975),
        "severe_error_rate": _percentile_interval(severe_samples, 0.025, 0.975),
    }


def write_decisions(path: str | Path, rows: list[DecisionRow]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "policy",
                "example_id",
                "domain",
                "user_profile",
                "correctness",
                "severe_if_wrong",
                "accept",
                "p_correct",
                "utility",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.policy,
                    r.example_id,
                    r.domain,
                    r.user_profile,
                    r.correctness,
                    r.severe_if_wrong,
                    r.accept,
                    f"{r.p_correct:.6f}",
                    f"{r.utility:.6f}",
                ]
            )


def write_metrics(path: str | Path, rows: list[MetricsRow]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "policy",
                "slice_name",
                "n",
                "utility_mean",
                "coverage",
                "escalation_rate",
                "accepted_accuracy",
                "severe_error_rate",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.policy,
                    r.slice_name,
                    r.n,
                    f"{r.utility_mean:.6f}",
                    f"{r.coverage:.6f}",
                    f"{r.escalation_rate:.6f}",
                    "" if r.accepted_accuracy is None else f"{r.accepted_accuracy:.6f}",
                    f"{r.severe_error_rate:.6f}",
                ]
            )


def _group(rows: list[DecisionRow], key_fn) -> dict[str, list[DecisionRow]]:
    grouped: dict[str, list[DecisionRow]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return grouped


def _percentile_interval(values: list[float], low_q: float, high_q: float) -> tuple[float, float]:
    ordered = sorted(values)
    if not ordered:
        return (0.0, 0.0)
    low_idx = int(low_q * (len(ordered) - 1))
    high_idx = int(high_q * (len(ordered) - 1))
    return (ordered[low_idx], ordered[high_idx])
