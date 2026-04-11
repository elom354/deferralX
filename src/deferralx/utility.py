from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from deferralx.schema import Record


@dataclass
class UtilityProfile:
    reward_correct: float
    penalty_wrong: float
    penalty_severe: float
    penalty_escalate: float


@dataclass
class UtilityConfig:
    profiles: dict[str, UtilityProfile]
    domain_risk_multiplier: dict[str, float]


DEFAULT_UTILITY_CONFIG = {
    "profiles": {
        "cautious_novice": {
            "reward_correct": 1.0,
            "penalty_wrong": -5.0,
            "penalty_severe": -20.0,
            "penalty_escalate": -0.60,
        },
        "balanced_user": {
            "reward_correct": 1.0,
            "penalty_wrong": -2.5,
            "penalty_severe": -12.0,
            "penalty_escalate": -0.40,
        },
        "expert_fast": {
            "reward_correct": 1.0,
            "penalty_wrong": -1.2,
            "penalty_severe": -6.0,
            "penalty_escalate": -0.20,
        },
    },
    "domain_risk_multiplier": {
        "general": 1.0,
        "finance": 1.6,
        "medical": 2.0,
    },
}


def load_utility_config(path: str | Path | None = None) -> UtilityConfig:
    raw = DEFAULT_UTILITY_CONFIG
    if path is not None:
        cfg_path = Path(path)
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)

    profiles: dict[str, UtilityProfile] = {}
    for name, payload in raw["profiles"].items():
        profiles[name.lower()] = UtilityProfile(
            reward_correct=float(payload["reward_correct"]),
            penalty_wrong=float(payload["penalty_wrong"]),
            penalty_severe=float(payload["penalty_severe"]),
            penalty_escalate=float(payload["penalty_escalate"]),
        )

    multipliers = {
        key.lower(): float(value) for key, value in raw["domain_risk_multiplier"].items()
    }
    return UtilityConfig(profiles=profiles, domain_risk_multiplier=multipliers)


def _domain_multiplier(record: Record, config: UtilityConfig) -> float:
    return config.domain_risk_multiplier.get(record.domain, 1.0)


def realized_utility(record: Record, accept: bool, config: UtilityConfig) -> float:
    profile = config.profiles[record.user_profile]
    if not accept:
        return profile.penalty_escalate

    if record.correctness == 1:
        return profile.reward_correct

    risk_mult = _domain_multiplier(record, config)
    utility = profile.penalty_wrong * risk_mult
    if record.severe_if_wrong == 1:
        utility += profile.penalty_severe * risk_mult
    return utility


def expected_utility_if_accept(
    record: Record, p_correct: float, config: UtilityConfig
) -> float:
    profile = config.profiles[record.user_profile]
    risk_mult = _domain_multiplier(record, config)

    wrong_utility = profile.penalty_wrong * risk_mult
    if record.severe_if_wrong == 1:
        wrong_utility += profile.penalty_severe * risk_mult

    return p_correct * profile.reward_correct + (1.0 - p_correct) * wrong_utility


def expected_utility_if_escalate(record: Record, config: UtilityConfig) -> float:
    return config.profiles[record.user_profile].penalty_escalate
