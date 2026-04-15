from __future__ import annotations

import argparse
import csv
from pathlib import Path

REQUIRED = [
    "example_id",
    "domain",
    "user_profile",
    "prompt",
    "reference_answer",
    "severe_if_wrong",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge DeferralX question CSV files")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input CSV paths")
    parser.add_argument("--output", required=True, help="Merged output CSV path")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    total = 0
    with out_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(REQUIRED)

        for p in args.inputs:
            in_path = Path(p)
            with in_path.open("r", encoding="utf-8", newline="") as f_in:
                reader = csv.DictReader(f_in)
                if reader.fieldnames is None:
                    continue
                missing = [c for c in REQUIRED if c not in reader.fieldnames]
                if missing:
                    raise ValueError(f"{in_path} missing columns: {missing}")

                for row in reader:
                    ex_id = str(row["example_id"]).strip()
                    if not ex_id:
                        continue
                    if ex_id in seen_ids:
                        continue
                    seen_ids.add(ex_id)
                    writer.writerow([row[c] for c in REQUIRED])
                    total += 1

    print(f"Merged {total} unique rows into {out_path}")


if __name__ == "__main__":
    main()
