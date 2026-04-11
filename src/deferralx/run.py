from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from deferralx.calibration import BinnedCalibrator
from deferralx.evaluation import (
    aggregate_by_slice,
    bootstrap_ci,
    run_policy,
    write_decisions,
    write_metrics,
)
from deferralx.policies import (
    AlwaysEscalatePolicy,
    GlobalThresholdPolicy,
    GroupThresholdPolicy,
    LinearUtilityRouterPolicy,
    RTRStyleThresholdRouterPolicy,
)
from deferralx.real_data import (
    build_client_from_env,
    build_local_hf_client,
    collect_real_records,
    load_question_records,
    save_real_records,
)
from deferralx.schema import load_records, stratified_split
from deferralx.synthetic import generate_synthetic_csv
from deferralx.utility import load_utility_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deferral experiments for user/task adaptive routing"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="Generate synthetic dataset")
    gen.add_argument("--output", default="data/synthetic_llm_logs.csv")
    gen.add_argument("--n", type=int, default=6000)
    gen.add_argument("--seed", type=int, default=42)

    exp = sub.add_parser("run", help="Run deferral policy experiments")
    exp.add_argument("--input", default="data/synthetic_llm_logs.csv")
    exp.add_argument("--outdir", default="outputs/main")
    exp.add_argument("--utility-config", default="configs/utility_config.json")
    exp.add_argument("--test-ratio", type=float, default=0.30)
    exp.add_argument("--seed", type=int, default=42)
    exp.add_argument("--bootstrap", type=int, default=400)
    exp.add_argument("--train-input", default="")
    exp.add_argument("--test-input", default="")
    exp.add_argument("--auto-generate", action="store_true")
    exp.add_argument("--auto-generate-n", type=int, default=6000)

    collect = sub.add_parser("collect-real", help="Collect real LLM logs from question CSV")
    collect.add_argument("--questions", required=True)
    collect.add_argument("--output", default="data/real_llm_logs.csv")
    collect.add_argument("--audit-jsonl", default="outputs/audit/real_collection.jsonl")
    collect.add_argument("--model", default="gpt-4.1-mini")
    collect.add_argument("--base-url", default="https://api.openai.com/v1")
    collect.add_argument("--api-key-env", default="OPENAI_API_KEY")
    collect.add_argument("--timeout-s", type=int, default=120)
    collect.add_argument("--max-tokens", type=int, default=256)
    collect.add_argument("--agreement-samples", type=int, default=3)
    collect.add_argument("--agreement-temperature", type=float, default=0.7)
    collect.add_argument("--fast-latency-s", type=float, default=4.0)
    collect.add_argument(
        "--system-prompt",
        default="You are a precise assistant. Provide your best concise answer.",
    )

    local_hf = sub.add_parser(
        "collect-local-hf",
        help="Collect real LLM logs from a local Hugging Face model",
    )
    local_hf.add_argument("--questions", required=True)
    local_hf.add_argument("--output", default="data/real_llm_logs_local.csv")
    local_hf.add_argument(
        "--audit-jsonl", default="outputs/audit/real_collection_local_hf.jsonl"
    )
    local_hf.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model id or local path",
    )
    local_hf.add_argument(
        "--device",
        default="auto",
        help="auto|cpu|mps|cuda",
    )
    local_hf.add_argument("--use-fp16", action="store_true")
    local_hf.add_argument("--max-tokens", type=int, default=256)
    local_hf.add_argument("--agreement-samples", type=int, default=3)
    local_hf.add_argument("--agreement-temperature", type=float, default=0.7)
    local_hf.add_argument("--fast-latency-s", type=float, default=4.0)
    local_hf.add_argument(
        "--system-prompt",
        default="You are a precise assistant. Provide your best concise answer.",
    )

    multi = sub.add_parser("run-multiseed", help="Run experiments for multiple seeds")
    multi.add_argument("--input", default="data/synthetic_llm_logs.csv")
    multi.add_argument("--outdir", default="outputs/multiseed")
    multi.add_argument("--utility-config", default="configs/utility_config.json")
    multi.add_argument("--test-ratio", type=float, default=0.30)
    multi.add_argument("--seeds", default="7,11,17,23,31")
    multi.add_argument("--bootstrap", type=int, default=120)
    multi.add_argument("--auto-generate", action="store_true")
    multi.add_argument("--auto-generate-n", type=int, default=6000)

    args = parser.parse_args()

    if args.cmd == "generate":
        generate_synthetic_csv(args.output, n=args.n, seed=args.seed)
        print(f"Synthetic dataset generated at: {args.output}")
        return

    if args.cmd == "collect-real":
        questions = load_question_records(args.questions)
        client = build_client_from_env(
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            timeout_s=args.timeout_s,
        )
        records = collect_real_records(
            questions=questions,
            client=client,
            model=args.model,
            max_tokens=args.max_tokens,
            agreement_samples=args.agreement_samples,
            agreement_temperature=args.agreement_temperature,
            fast_latency_s=args.fast_latency_s,
            system_prompt=args.system_prompt,
            audit_path=args.audit_jsonl,
        )
        save_real_records(args.output, records)
        print(f"Real LLM records written to: {args.output}")
        return

    if args.cmd == "collect-local-hf":
        questions = load_question_records(args.questions)
        client = build_local_hf_client(
            model_id_or_path=args.model_id,
            device=args.device,
            use_fp16=args.use_fp16,
        )
        records = collect_real_records(
            questions=questions,
            client=client,
            model=args.model_id,
            max_tokens=args.max_tokens,
            agreement_samples=args.agreement_samples,
            agreement_temperature=args.agreement_temperature,
            fast_latency_s=args.fast_latency_s,
            system_prompt=args.system_prompt,
            audit_path=args.audit_jsonl,
        )
        save_real_records(args.output, records)
        print(f"Local HF LLM records written to: {args.output}")
        return

    input_path = Path(args.input)
    if input_path.exists() is False and args.auto_generate:
        gen_seed = args.seed if hasattr(args, "seed") else 42
        generate_synthetic_csv(str(input_path), n=args.auto_generate_n, seed=gen_seed)
        print(f"Input dataset not found. Generated synthetic data: {input_path}")

    if args.cmd == "run-multiseed":
        seed_values = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        run_multiseed(
            input_path=str(input_path),
            outdir=args.outdir,
            utility_config_path=args.utility_config,
            test_ratio=args.test_ratio,
            seeds=seed_values,
            n_boot=args.bootstrap,
        )
        return

    if bool(args.train_input.strip()) != bool(args.test_input.strip()):
        raise ValueError("Provide both --train-input and --test-input, or neither.")

    run_experiment(
        input_path=str(input_path),
        outdir=args.outdir,
        utility_config_path=args.utility_config,
        test_ratio=args.test_ratio,
        seed=args.seed,
        n_boot=args.bootstrap,
        train_input=args.train_input,
        test_input=args.test_input,
    )


def run_experiment(
    input_path: str,
    outdir: str,
    utility_config_path: str,
    test_ratio: float,
    seed: int,
    n_boot: int,
    train_input: str = "",
    test_input: str = "",
) -> None:
    utility_config = load_utility_config(utility_config_path)
    has_fixed_split = bool(train_input.strip()) and bool(test_input.strip())

    if has_fixed_split:
        train = load_records(train_input)
        test = load_records(test_input)
        records = train + test
    else:
        records = load_records(input_path)
        train, test = stratified_split(records, test_ratio=test_ratio, seed=seed)

    calibrator = BinnedCalibrator(n_bins=15, laplace=1.0)
    train_conf = [r.base_confidence for r in train]
    train_labels = [r.correctness for r in train]
    calibrator.fit(train_conf, train_labels)

    for r in train:
        r.calibrated_confidence = calibrator.predict_one(r.base_confidence)
    for r in test:
        r.calibrated_confidence = calibrator.predict_one(r.base_confidence)

    policies = [
        AlwaysEscalatePolicy(),
        GlobalThresholdPolicy(step=0.01),
        GroupThresholdPolicy(mode="domain", step=0.01),
        GroupThresholdPolicy(mode="profile", step=0.01),
        GroupThresholdPolicy(mode="domain_profile", step=0.01),
        RTRStyleThresholdRouterPolicy(step=0.01),
        LinearUtilityRouterPolicy(learning_rate=0.12, epochs=500, l2=0.001),
    ]

    out_root = Path(outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_overall_metrics = []
    all_domain_metrics = []
    all_profile_metrics = []
    all_domain_profile_metrics = []

    ci_rows = []
    threshold_dump = {}

    for policy in policies:
        policy.fit(train, utility_config)
        decisions = run_policy(policy, test, utility_config)

        write_decisions(out_root / f"decisions_{policy.name}.csv", decisions)

        metrics = aggregate_by_slice(policy.name, decisions)
        all_overall_metrics.extend(metrics["overall"])
        all_domain_metrics.extend(metrics["domain"])
        all_profile_metrics.extend(metrics["profile"])
        all_domain_profile_metrics.extend(metrics["domain_profile"])

        ci = bootstrap_ci(decisions, n_boot=n_boot, seed=seed)
        ci_rows.append(
            {
                "policy": policy.name,
                "utility_mean_ci_low": ci["utility_mean"][0],
                "utility_mean_ci_high": ci["utility_mean"][1],
                "coverage_ci_low": ci["coverage"][0],
                "coverage_ci_high": ci["coverage"][1],
                "severe_error_rate_ci_low": ci["severe_error_rate"][0],
                "severe_error_rate_ci_high": ci["severe_error_rate"][1],
            }
        )

        if hasattr(policy, "threshold"):
            threshold_dump[policy.name] = getattr(policy, "threshold")
        if hasattr(policy, "thresholds"):
            threshold_dump[policy.name] = getattr(policy, "thresholds")

    write_metrics(out_root / "metrics_overall.csv", all_overall_metrics)
    write_metrics(out_root / "metrics_by_domain.csv", all_domain_metrics)
    write_metrics(out_root / "metrics_by_profile.csv", all_profile_metrics)
    write_metrics(out_root / "metrics_by_domain_profile.csv", all_domain_profile_metrics)

    _write_ci_csv(out_root / "metrics_overall_ci.csv", ci_rows)
    _write_rankings(
        out_root / "policy_rankings.csv",
        all_overall_metrics,
        all_domain_metrics,
        all_profile_metrics,
        all_domain_profile_metrics,
    )

    with (out_root / "selected_thresholds.json").open("w", encoding="utf-8") as f:
        json.dump(threshold_dump, f, indent=2)

    _write_summary_md(
        out_root / "summary.md",
        total_n=len(records),
        train_n=len(train),
        test_n=len(test),
        overall_metrics=all_overall_metrics,
        domain_metrics=all_domain_metrics,
        profile_metrics=all_profile_metrics,
    )

    print("Experiment completed.")
    print(f"Results written to: {out_root}")


def run_multiseed(
    input_path: str,
    outdir: str,
    utility_config_path: str,
    test_ratio: float,
    seeds: list[int],
    n_boot: int,
) -> None:
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)

    policy_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    for seed in seeds:
        run_experiment(
            input_path=input_path,
            outdir=str(root / f"seed_{seed}"),
            utility_config_path=utility_config_path,
            test_ratio=test_ratio,
            seed=seed,
            n_boot=n_boot,
            train_input="",
            test_input="",
        )
        metrics = _read_overall_metrics(root / f"seed_{seed}" / "metrics_overall.csv")
        for row in metrics:
            policy_rows[row["policy"]].append(row)

    summary_path = root / "multiseed_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "policy",
                "num_seeds",
                "utility_mean_avg",
                "utility_mean_std",
                "coverage_avg",
                "coverage_std",
                "severe_error_rate_avg",
                "severe_error_rate_std",
            ]
        )
        for policy, rows in sorted(policy_rows.items()):
            writer.writerow(
                [
                    policy,
                    len(rows),
                    f"{_mean([r['utility_mean'] for r in rows]):.6f}",
                    f"{_std([r['utility_mean'] for r in rows]):.6f}",
                    f"{_mean([r['coverage'] for r in rows]):.6f}",
                    f"{_std([r['coverage'] for r in rows]):.6f}",
                    f"{_mean([r['severe_error_rate'] for r in rows]):.6f}",
                    f"{_std([r['severe_error_rate'] for r in rows]):.6f}",
                ]
            )

    print("Multiseed experiment completed.")
    print(f"Results written to: {root}")


def _read_overall_metrics(path: Path) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                {
                    "policy": row["policy"],
                    "utility_mean": float(row["utility_mean"]),
                    "coverage": float(row["coverage"]),
                    "severe_error_rate": float(row["severe_error_rate"]),
                }
            )
    return out


def _write_ci_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "policy",
                "utility_mean_ci_low",
                "utility_mean_ci_high",
                "coverage_ci_low",
                "coverage_ci_high",
                "severe_error_rate_ci_low",
                "severe_error_rate_ci_high",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["policy"],
                    f"{row['utility_mean_ci_low']:.6f}",
                    f"{row['utility_mean_ci_high']:.6f}",
                    f"{row['coverage_ci_low']:.6f}",
                    f"{row['coverage_ci_high']:.6f}",
                    f"{row['severe_error_rate_ci_low']:.6f}",
                    f"{row['severe_error_rate_ci_high']:.6f}",
                ]
            )


def _write_rankings(
    path: Path,
    overall_metrics,
    domain_metrics,
    profile_metrics,
    domain_profile_metrics,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["grouping", "slice_name", "rank", "policy", "utility_mean"])

        for grouping, rows in [
            ("overall", overall_metrics),
            ("domain", domain_metrics),
            ("profile", profile_metrics),
            ("domain_profile", domain_profile_metrics),
        ]:
            per_slice: dict[str, list] = defaultdict(list)
            for row in rows:
                per_slice[row.slice_name].append(row)

            for slice_name, slice_rows in sorted(per_slice.items()):
                sorted_rows = sorted(slice_rows, key=lambda r: r.utility_mean, reverse=True)
                rank = 1
                for r in sorted_rows:
                    writer.writerow([grouping, slice_name, rank, r.policy, f"{r.utility_mean:.6f}"])
                    rank += 1


def _write_summary_md(
    path: Path,
    total_n: int,
    train_n: int,
    test_n: int,
    overall_metrics,
    domain_metrics,
    profile_metrics,
) -> None:
    overall_sorted = sorted(overall_metrics, key=lambda r: r.utility_mean, reverse=True)
    best_overall = overall_sorted[0]

    top_by_domain = _best_by_slice(domain_metrics)
    top_by_profile = _best_by_slice(profile_metrics)

    overall_best_policy = best_overall.policy
    domain_policy_set = {v[0] for v in top_by_domain.values()}
    profile_policy_set = {v[0] for v in top_by_profile.values()}

    with path.open("w", encoding="utf-8") as f:
        f.write("# DeferralX Experiment Summary\n\n")
        f.write(f"- Total samples: {total_n}\n")
        f.write(f"- Train samples: {train_n}\n")
        f.write(f"- Test samples: {test_n}\n\n")

        f.write("## Best Overall Policy\n\n")
        f.write(
            f"- {best_overall.policy}: utility={best_overall.utility_mean:.4f}, coverage={best_overall.coverage:.4f}, "
            f"accepted_accuracy={_fmt_optional(best_overall.accepted_accuracy)}, severe_error_rate={best_overall.severe_error_rate:.4f}\n\n"
        )

        f.write("## Best Policy by Domain\n\n")
        for slice_name, (policy, utility) in sorted(top_by_domain.items()):
            f.write(f"- {slice_name}: {policy} (utility={utility:.4f})\n")
        f.write("\n")

        f.write("## Best Policy by User Profile\n\n")
        for slice_name, (policy, utility) in sorted(top_by_profile.items()):
            f.write(f"- {slice_name}: {policy} (utility={utility:.4f})\n")
        f.write("\n")

        f.write("## Heterogeneity Check\n\n")
        f.write(f"- Overall best policy: {overall_best_policy}\n")
        f.write(f"- Distinct best policies across domains: {sorted(domain_policy_set)}\n")
        f.write(f"- Distinct best policies across profiles: {sorted(profile_policy_set)}\n")


def _best_by_slice(rows) -> dict[str, tuple[str, float]]:
    grouped: dict[str, list] = defaultdict(list)
    for row in rows:
        grouped[row.slice_name].append(row)

    out: dict[str, tuple[str, float]] = {}
    for slice_name, slice_rows in grouped.items():
        best = sorted(slice_rows, key=lambda r: r.utility_mean, reverse=True)[0]
        out[slice_name] = (best.policy, best.utility_mean)
    return out


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def _mean(values: list[float]) -> float:
    if len(values) == 0:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((x - mean) * (x - mean) for x in values) / (len(values) - 1)
    return variance ** 0.5


if __name__ == "__main__":
    main()
