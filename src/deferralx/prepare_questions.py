from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DeferralX question CSV from online Hugging Face datasets"
    )
    parser.add_argument("--dataset", required=True, help="HF dataset id, e.g. cais/mmlu")
    parser.add_argument("--split", default="test", help="Dataset split to load")
    parser.add_argument(
        "--subset",
        default="",
        help="Optional dataset subset/config name",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path (DeferralX question schema)",
    )
    parser.add_argument(
        "--question-col",
        required=True,
        help="Column name containing question text",
    )
    parser.add_argument(
        "--answer-col",
        required=True,
        help="Column name containing reference answer",
    )
    parser.add_argument(
        "--id-col",
        default="",
        help="Optional id column. If empty, auto ids are created.",
    )
    parser.add_argument(
        "--choices-col",
        default="",
        help="Optional choices column for MCQ. Choices are appended to prompt.",
    )
    parser.add_argument(
        "--answer-is-index",
        action="store_true",
        help="Interpret answer-col as index in choices-col.",
    )
    parser.add_argument(
        "--domain",
        required=True,
        choices=["general", "finance", "medical"],
    )
    parser.add_argument(
        "--domain-mode",
        default="fixed",
        choices=["fixed", "mmlu_subject"],
        help="Domain assignment mode. mmlu_subject maps MMLU `subject` to {general,finance,medical}.",
    )
    parser.add_argument(
        "--default-profile",
        default="balanced_user",
        choices=["cautious_novice", "balanced_user", "expert_fast"],
    )
    parser.add_argument(
        "--profile-mode",
        default="fixed",
        choices=["fixed", "cycle", "random"],
        help="How to assign user_profile in output rows.",
    )
    parser.add_argument(
        "--profile-seed",
        type=int,
        default=42,
        help="Random seed used when --profile-mode=random.",
    )
    parser.add_argument(
        "--severe-if-wrong",
        type=int,
        default=1,
        choices=[0, 1],
        help="Default severe_if_wrong label for exported rows.",
    )
    parser.add_argument(
        "--severe-mode",
        default="fixed",
        choices=["fixed", "by_domain"],
        help="Severity assignment mode. by_domain sets severe=1 for finance/medical and 0 for general.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of rows (0 = all)",
    )
    parser.add_argument(
        "--print-columns",
        action="store_true",
        help="Print dataset columns and exit.",
    )

    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "This command requires the `datasets` package. Install with: pip install datasets"
        ) from e

    try:
        if args.subset.strip():
            ds = load_dataset(args.dataset, args.subset, split=args.split)
        else:
            ds = load_dataset(args.dataset, split=args.split)
    except ValueError as e:
        message = str(e)
        if "Config name is missing" in message:
            raise ValueError(
                f"{message}\n\nTip: add --subset. For MMLU, a common choice is "
                "`--subset all` or a subject like `--subset abstract_algebra`."
            ) from e
        raise

    if args.print_columns:
        print("Columns:", list(ds.column_names))
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "example_id",
                "domain",
                "user_profile",
                "prompt",
                "reference_answer",
                "severe_if_wrong",
            ]
        )

        n_written = 0
        rng = random.Random(args.profile_seed)
        cycle_profiles = ["cautious_novice", "balanced_user", "expert_fast"]
        for i, row in enumerate(ds):
            if args.limit > 0 and n_written >= args.limit:
                break

            question = _safe_text(row.get(args.question_col, ""))
            if not question:
                continue

            prompt = question
            answer_value = row.get(args.answer_col, "")

            if args.choices_col:
                choices = row.get(args.choices_col)
                choice_list = _normalize_choices(choices)
                if choice_list:
                    prompt = _append_choices_to_prompt(question, choice_list)
                    if args.answer_is_index:
                        answer_value = _answer_from_index(answer_value, choice_list)

            reference_answer = _safe_text(answer_value)
            if not reference_answer:
                continue

            if args.id_col:
                example_id = _safe_text(row.get(args.id_col, ""))
                if not example_id:
                    example_id = f"{args.domain}_{i:06d}"
            else:
                example_id = f"{args.domain}_{i:06d}"

            user_profile = _select_profile(
                mode=args.profile_mode,
                default_profile=args.default_profile,
                cycle_profiles=cycle_profiles,
                written_index=n_written,
                rng=rng,
            )
            domain = _select_domain(
                mode=args.domain_mode,
                default_domain=args.domain,
                row=row,
            )

            writer.writerow(
                [
                    example_id,
                    domain,
                    user_profile,
                    prompt,
                    reference_answer,
                    _select_severity(
                        mode=args.severe_mode,
                        default_value=args.severe_if_wrong,
                        domain=domain,
                    ),
                ]
            )
            n_written += 1

    print(f"Wrote {n_written} rows to {out_path}")


def _safe_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def _normalize_choices(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_safe_text(x) for x in value if _safe_text(x)]
    if isinstance(value, dict):
        items = []
        for key in sorted(value.keys()):
            t = _safe_text(value[key])
            if t:
                items.append(t)
        return items
    txt = _safe_text(value)
    if not txt:
        return []
    return [txt]


def _append_choices_to_prompt(question: str, choices: list[str]) -> str:
    letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    lines = [question, "", "Options:"]
    for idx, choice in enumerate(choices):
        label = letters[idx] if idx < len(letters) else f"OPT{idx+1}"
        lines.append(f"{label}. {choice}")
    return "\n".join(lines)


def _answer_from_index(answer_value, choices: list[str]) -> str:
    try:
        idx = int(answer_value)
    except Exception:
        return _safe_text(answer_value)

    if idx < 0 or idx >= len(choices):
        return _safe_text(answer_value)
    return _safe_text(choices[idx])


def _select_profile(
    mode: str,
    default_profile: str,
    cycle_profiles: list[str],
    written_index: int,
    rng: random.Random,
) -> str:
    if mode == "fixed":
        return default_profile
    if mode == "cycle":
        return cycle_profiles[written_index % len(cycle_profiles)]
    # random mode
    return rng.choice(cycle_profiles)


def _select_domain(mode: str, default_domain: str, row) -> str:
    if mode == "fixed":
        return default_domain

    subject = _safe_text(row.get("subject", "")).lower()
    finance_subjects = {
        "econometrics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "professional_accounting",
        "business_ethics",
        "management",
        "marketing",
    }
    medical_subjects = {
        "anatomy",
        "clinical_knowledge",
        "college_medicine",
        "human_aging",
        "medical_genetics",
        "nutrition",
        "professional_medicine",
        "virology",
    }

    if subject in finance_subjects:
        return "finance"
    if subject in medical_subjects:
        return "medical"
    return "general"


def _select_severity(mode: str, default_value: int, domain: str) -> int:
    if mode == "fixed":
        return int(default_value)
    return 0 if domain == "general" else 1


if __name__ == "__main__":
    main()
