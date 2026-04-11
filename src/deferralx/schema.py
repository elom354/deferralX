from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REQUIRED_COLUMNS = {
    "example_id",
    "domain",
    "user_profile",
    "correctness",
    "severe_if_wrong",
    "p_internal",
    "p_verbal",
    "agreement",
    "response_speed",
}


@dataclass
class Record:
    example_id: str
    domain: str
    user_profile: str
    correctness: int
    severe_if_wrong: int
    p_internal: float
    p_verbal: float
    agreement: float
    response_speed: str
    calibrated_confidence: float = 0.0

    @property
    def speed_fast(self) -> float:
        return 1.0 if self.response_speed.strip().lower() == "fast" else 0.0

    @property
    def base_confidence(self) -> float:
        # Weighted blend motivated by calibration literature:
        # internal prob + verbalized confidence + agreement/self-consistency.
        value = 0.50 * self.p_internal + 0.30 * self.p_verbal + 0.20 * self.agreement
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value


def _parse_int_flag(value: str) -> int:
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return 1
    if text in {"0", "false", "no", "n"}:
        return 0
    raise ValueError(f"Invalid binary value: {value!r}")


def _parse_prob(value: str) -> float:
    prob = float(value)
    if prob < 0.0:
        return 0.0
    if prob > 1.0:
        return 1.0
    return prob


def load_records(path: str | Path) -> list[Record]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    records: list[Record] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            record = Record(
                example_id=row["example_id"],
                domain=row["domain"].strip().lower(),
                user_profile=row["user_profile"].strip().lower(),
                correctness=_parse_int_flag(row["correctness"]),
                severe_if_wrong=_parse_int_flag(row["severe_if_wrong"]),
                p_internal=_parse_prob(row["p_internal"]),
                p_verbal=_parse_prob(row["p_verbal"]),
                agreement=_parse_prob(row["agreement"]),
                response_speed=row["response_speed"].strip().lower(),
            )
            records.append(record)

    if not records:
        raise ValueError("Dataset is empty")
    return records


def save_records(path: str | Path, records: Iterable[Record]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "example_id",
                "domain",
                "user_profile",
                "correctness",
                "severe_if_wrong",
                "p_internal",
                "p_verbal",
                "agreement",
                "response_speed",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r.example_id,
                    r.domain,
                    r.user_profile,
                    r.correctness,
                    r.severe_if_wrong,
                    f"{r.p_internal:.6f}",
                    f"{r.p_verbal:.6f}",
                    f"{r.agreement:.6f}",
                    r.response_speed,
                ]
            )


def stratified_split(
    records: list[Record], test_ratio: float = 0.3, seed: int = 42
) -> tuple[list[Record], list[Record]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")

    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for r in records:
        groups[(r.domain, r.user_profile)].append(r)

    train: list[Record] = []
    test: list[Record] = []
    for group in groups.values():
        shuffled = list(group)
        rng.shuffle(shuffled)

        n_test = int(len(shuffled) * test_ratio)
        if len(shuffled) >= 4:
            n_test = max(1, n_test)
            n_test = min(n_test, len(shuffled) - 1)

        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test
