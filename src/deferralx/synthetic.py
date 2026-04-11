from __future__ import annotations

import random

from deferralx.schema import Record, save_records


def generate_synthetic_records(n: int = 6000, seed: int = 42) -> list[Record]:
    rng = random.Random(seed)

    domain_specs = {
        "general": {
            "weight": 0.45,
            "base_acc": 0.78,
            "difficulty_beta": (2.0, 4.0),
            "severe_rate": 0.07,
            "overconfidence": 0.08,
        },
        "finance": {
            "weight": 0.28,
            "base_acc": 0.63,
            "difficulty_beta": (2.5, 2.5),
            "severe_rate": 0.40,
            "overconfidence": 0.12,
        },
        "medical": {
            "weight": 0.27,
            "base_acc": 0.58,
            "difficulty_beta": (3.0, 2.2),
            "severe_rate": 0.52,
            "overconfidence": 0.14,
        },
    }

    profile_specs = {
        "cautious_novice": {
            "weight": 0.35,
            "speed_fast_prob": 0.20,
            "accuracy_adj": -0.02,
            "verbal_bias": -0.03,
        },
        "balanced_user": {
            "weight": 0.40,
            "speed_fast_prob": 0.45,
            "accuracy_adj": 0.00,
            "verbal_bias": 0.00,
        },
        "expert_fast": {
            "weight": 0.25,
            "speed_fast_prob": 0.75,
            "accuracy_adj": 0.03,
            "verbal_bias": 0.04,
        },
    }

    domain_keys = list(domain_specs.keys())
    domain_weights = [domain_specs[d]["weight"] for d in domain_keys]

    profile_keys = list(profile_specs.keys())
    profile_weights = [profile_specs[p]["weight"] for p in profile_keys]

    rows: list[Record] = []
    for i in range(n):
        domain = rng.choices(domain_keys, weights=domain_weights, k=1)[0]
        profile = rng.choices(profile_keys, weights=profile_weights, k=1)[0]

        domain_info = domain_specs[domain]
        profile_info = profile_specs[profile]

        a, b = domain_info["difficulty_beta"]
        difficulty = rng.betavariate(a, b)

        p_correct = (
            domain_info["base_acc"]
            + profile_info["accuracy_adj"]
            - 0.50 * difficulty
            + rng.uniform(-0.05, 0.05)
        )
        p_correct = _clamp(p_correct)

        correctness = 1 if rng.random() < p_correct else 0

        severe_if_wrong = 1 if rng.random() < domain_info["severe_rate"] else 0

        # Simulated confidence channels with realistic miscalibration.
        p_internal = _clamp(
            p_correct
            + domain_info["overconfidence"]
            + rng.uniform(-0.18, 0.18)
            + (0.03 if correctness == 1 else -0.02)
        )

        p_verbal = _clamp(
            p_internal
            + profile_info["verbal_bias"]
            + rng.uniform(-0.15, 0.15)
            + (0.02 if correctness == 1 else -0.02)
        )

        agreement = _clamp(
            0.20 + 0.70 * p_correct + rng.uniform(-0.12, 0.12) + (0.03 if correctness == 1 else 0.0)
        )

        response_speed = "fast" if rng.random() < profile_info["speed_fast_prob"] else "careful"

        rows.append(
            Record(
                example_id=f"ex_{i:06d}",
                domain=domain,
                user_profile=profile,
                correctness=correctness,
                severe_if_wrong=severe_if_wrong,
                p_internal=p_internal,
                p_verbal=p_verbal,
                agreement=agreement,
                response_speed=response_speed,
            )
        )

    return rows


def generate_synthetic_csv(path: str, n: int = 6000, seed: int = 42) -> None:
    save_records(path, generate_synthetic_records(n=n, seed=seed))


def _clamp(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
