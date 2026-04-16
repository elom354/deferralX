from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PolicyStats:
    policy: str
    utility_mean_avg: float
    utility_mean_std: float
    coverage_avg: float
    coverage_std: float
    severe_error_rate_avg: float
    severe_error_rate_std: float
    num_seeds: int


def report_multiseed(
    multiseed_dir: str | Path,
    outdir: str | Path,
    label: str,
) -> dict[str, str]:
    root = Path(multiseed_dir)
    out_root = Path(outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_stats = _read_multiseed_summary(root / "multiseed_summary.csv")
    if not summary_stats:
        raise ValueError(f"No policies found in {root / 'multiseed_summary.csv'}")

    seed_dirs = _list_seed_dirs(root)
    per_seed_overall = _read_per_seed_overall(seed_dirs)
    best_policy_counts = _count_best_policies(per_seed_overall)
    best_domain_counts = _count_best_by_slice(seed_dirs, "metrics_by_domain.csv")
    best_profile_counts = _count_best_by_slice(seed_dirs, "metrics_by_profile.csv")

    rows_sorted = sorted(summary_stats.values(), key=lambda x: x.utility_mean_avg, reverse=True)
    baseline = summary_stats.get("always_escalate")
    best = rows_sorted[0]
    gain_vs_baseline = (
        best.utility_mean_avg - baseline.utility_mean_avg if baseline is not None else 0.0
    )
    gain_by_seed = (
        _compute_gain_by_seed(
            per_seed_overall=per_seed_overall,
            policy=best.policy,
            baseline_policy=baseline.policy,
        )
        if baseline is not None
        else []
    )
    gain_vs_baseline_std = _std(gain_by_seed)
    gain_vs_baseline_min = min(gain_by_seed) if gain_by_seed else 0.0
    gain_vs_baseline_max = max(gain_by_seed) if gain_by_seed else 0.0

    summary_csv = out_root / "model_policy_summary.csv"
    _write_policy_summary_csv(summary_csv, rows_sorted)

    seed_winners_csv = out_root / "seed_winner_counts.csv"
    _write_counts_csv(seed_winners_csv, "policy", best_policy_counts)

    domain_winners_csv = out_root / "domain_winner_counts.csv"
    _write_slice_counts_csv(domain_winners_csv, "domain", best_domain_counts)

    profile_winners_csv = out_root / "profile_winner_counts.csv"
    _write_slice_counts_csv(profile_winners_csv, "profile", best_profile_counts)

    curve_csv = out_root / "utility_coverage_curve.csv"
    _write_curve_csv(curve_csv, rows_sorted)

    plot_path = out_root / "utility_coverage_curve.png"
    _plot_utility_coverage_curve(rows_sorted, label=label, path=plot_path)

    md_path = out_root / "results_section.md"
    _write_results_markdown(
        path=md_path,
        label=label,
        seed_dirs=seed_dirs,
        rows_sorted=rows_sorted,
        baseline=baseline,
        best=best,
        gain_vs_baseline=gain_vs_baseline,
        gain_vs_baseline_std=gain_vs_baseline_std,
        gain_vs_baseline_min=gain_vs_baseline_min,
        gain_vs_baseline_max=gain_vs_baseline_max,
        best_policy_counts=best_policy_counts,
        best_domain_counts=best_domain_counts,
        best_profile_counts=best_profile_counts,
        plot_path=plot_path,
    )

    return {
        "results_markdown": str(md_path),
        "policy_summary_csv": str(summary_csv),
        "seed_winner_counts_csv": str(seed_winners_csv),
        "domain_winner_counts_csv": str(domain_winners_csv),
        "profile_winner_counts_csv": str(profile_winners_csv),
        "utility_coverage_curve_csv": str(curve_csv),
        "utility_coverage_curve_png": str(plot_path),
    }


def compare_models(
    runs: list[tuple[str, str | Path]],
    outdir: str | Path,
) -> dict[str, str]:
    if len(runs) < 2:
        raise ValueError("compare-models requires at least two model runs")

    out_root = Path(outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    per_model: dict[str, list[PolicyStats]] = {}
    for label, path in runs:
        summary_stats = _read_multiseed_summary(Path(path) / "multiseed_summary.csv")
        if not summary_stats:
            raise ValueError(f"No policies found in {Path(path) / 'multiseed_summary.csv'}")
        rows = sorted(summary_stats.values(), key=lambda x: x.utility_mean_avg, reverse=True)
        per_model[label] = rows

    compare_csv = out_root / "model_comparison_summary.csv"
    with compare_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_label",
                "policy",
                "utility_mean_avg",
                "utility_mean_std",
                "coverage_avg",
                "coverage_std",
                "severe_error_rate_avg",
                "severe_error_rate_std",
                "num_seeds",
            ]
        )
        for label, rows in per_model.items():
            for row in rows:
                writer.writerow(
                    [
                        label,
                        row.policy,
                        f"{row.utility_mean_avg:.6f}",
                        f"{row.utility_mean_std:.6f}",
                        f"{row.coverage_avg:.6f}",
                        f"{row.coverage_std:.6f}",
                        f"{row.severe_error_rate_avg:.6f}",
                        f"{row.severe_error_rate_std:.6f}",
                        row.num_seeds,
                    ]
                )

    curve_png = out_root / "model_utility_coverage_curves.png"
    _plot_multi_model_curves(per_model, curve_png)

    md_path = out_root / "model_comparison_results.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Cross-Model Deferral Comparison\n\n")
        f.write(
            "This section is auto-generated from real multi-seed outputs. "
            "No synthetic data or inferred metrics are used.\n\n"
        )
        f.write("## Best Policy per Model (mean over seeds)\n\n")
        for label, rows in per_model.items():
            best = rows[0]
            baseline = next((r for r in rows if r.policy == "always_escalate"), None)
            gain = (
                best.utility_mean_avg - baseline.utility_mean_avg
                if baseline is not None
                else 0.0
            )
            f.write(
                f"- **{label}**: best policy `{best.policy}` "
                f"(utility={best.utility_mean_avg:.4f}, coverage={best.coverage_avg:.4f}, "
                f"severe_error_rate={best.severe_error_rate_avg:.4f}, gain_vs_always={gain:.4f})\n"
            )
        f.write("\n")
        f.write("## Utility-Coverage Curves\n\n")
        f.write(
            f"- Plot: `{curve_png}` (each line connects policy points sorted by coverage for one model)\n"
        )
        f.write(
            f"- Detailed table for all policies and models: `{compare_csv}`\n"
        )

    return {
        "comparison_markdown": str(md_path),
        "comparison_csv": str(compare_csv),
        "comparison_curve_png": str(curve_png),
    }


def _read_multiseed_summary(path: Path) -> dict[str, PolicyStats]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    out: dict[str, PolicyStats] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["policy"]] = PolicyStats(
                policy=row["policy"],
                utility_mean_avg=float(row["utility_mean_avg"]),
                utility_mean_std=float(row["utility_mean_std"]),
                coverage_avg=float(row["coverage_avg"]),
                coverage_std=float(row["coverage_std"]),
                severe_error_rate_avg=float(row["severe_error_rate_avg"]),
                severe_error_rate_std=float(row["severe_error_rate_std"]),
                num_seeds=int(row["num_seeds"]),
            )
    return out


def _list_seed_dirs(root: Path) -> list[Path]:
    dirs = [d for d in root.glob("seed_*") if d.is_dir()]
    dirs.sort(key=lambda p: p.name)
    return dirs


def _read_per_seed_overall(seed_dirs: list[Path]) -> dict[str, list[dict[str, float]]]:
    out: dict[str, list[dict[str, float]]] = {}
    for seed_dir in seed_dirs:
        metrics_path = seed_dir / "metrics_overall.csv"
        with metrics_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                rows.append(
                    {
                        "policy": row["policy"],
                        "utility_mean": float(row["utility_mean"]),
                    }
                )
            out[seed_dir.name] = rows
    return out


def _count_best_policies(
    per_seed_overall: dict[str, list[dict[str, float]]],
    tol: float = 1e-12,
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for rows in per_seed_overall.values():
        if not rows:
            continue
        best_u = max(r["utility_mean"] for r in rows)
        for r in rows:
            if abs(r["utility_mean"] - best_u) <= tol:
                counts[r["policy"]] += 1
    return dict(sorted(counts.items()))


def _compute_gain_by_seed(
    per_seed_overall: dict[str, list[dict[str, float]]],
    policy: str,
    baseline_policy: str,
) -> list[float]:
    gains: list[float] = []
    for rows in per_seed_overall.values():
        by_policy = {r["policy"]: r["utility_mean"] for r in rows}
        if policy not in by_policy or baseline_policy not in by_policy:
            continue
        gains.append(by_policy[policy] - by_policy[baseline_policy])
    return gains


def _count_best_by_slice(
    seed_dirs: list[Path],
    filename: str,
    tol: float = 1e-12,
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for seed_dir in seed_dirs:
        path = seed_dir / filename
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in reader:
                grouped[row["slice_name"]].append(row)
            for slice_name, rows in grouped.items():
                best_u = max(float(r["utility_mean"]) for r in rows)
                for r in rows:
                    if abs(float(r["utility_mean"]) - best_u) <= tol:
                        out[slice_name][r["policy"]] += 1
    return {
        k: dict(sorted(v.items()))
        for k, v in sorted(out.items(), key=lambda x: x[0])
    }


def _write_policy_summary_csv(path: Path, rows: list[PolicyStats]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "policy",
                "utility_mean_avg",
                "utility_mean_std",
                "coverage_avg",
                "coverage_std",
                "severe_error_rate_avg",
                "severe_error_rate_std",
                "num_seeds",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.policy,
                    f"{r.utility_mean_avg:.6f}",
                    f"{r.utility_mean_std:.6f}",
                    f"{r.coverage_avg:.6f}",
                    f"{r.coverage_std:.6f}",
                    f"{r.severe_error_rate_avg:.6f}",
                    f"{r.severe_error_rate_std:.6f}",
                    r.num_seeds,
                ]
            )


def _write_counts_csv(path: Path, key_name: str, counts: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([key_name, "win_count"])
        for key, value in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            writer.writerow([key, value])


def _write_slice_counts_csv(
    path: Path,
    slice_name: str,
    counts: dict[str, dict[str, int]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([slice_name, "policy", "win_count"])
        for slice_value, policy_counts in counts.items():
            for policy, value in sorted(policy_counts.items(), key=lambda x: (-x[1], x[0])):
                writer.writerow([slice_value, policy, value])


def _write_curve_csv(path: Path, rows: list[PolicyStats]) -> None:
    ordered = sorted(rows, key=lambda r: r.coverage_avg)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "coverage_avg", "utility_mean_avg", "severe_error_rate_avg"])
        for r in ordered:
            writer.writerow(
                [
                    r.policy,
                    f"{r.coverage_avg:.6f}",
                    f"{r.utility_mean_avg:.6f}",
                    f"{r.severe_error_rate_avg:.6f}",
                ]
            )


def _plot_utility_coverage_curve(rows: list[PolicyStats], label: str, path: Path) -> None:
    try:
        _ensure_matplotlib_cache()
        import matplotlib.pyplot as plt
    except Exception:
        return

    ordered = sorted(rows, key=lambda r: r.coverage_avg)
    x = [r.coverage_avg for r in ordered]
    y = [r.utility_mean_avg for r in ordered]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, marker="o", linewidth=1.5, label=label)
    for r in ordered:
        ax.annotate(r.policy, (r.coverage_avg, r.utility_mean_avg), fontsize=7)

    ax.set_xlabel("Coverage (avg across seeds)")
    ax.set_ylabel("Utility mean (avg across seeds)")
    ax.set_title(f"Utility-Coverage Curve: {label}")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_multi_model_curves(per_model: dict[str, list[PolicyStats]], path: Path) -> None:
    try:
        _ensure_matplotlib_cache()
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for label, rows in per_model.items():
        ordered = sorted(rows, key=lambda r: r.coverage_avg)
        x = [r.coverage_avg for r in ordered]
        y = [r.utility_mean_avg for r in ordered]
        ax.plot(x, y, marker="o", linewidth=1.4, label=label)
        best = max(rows, key=lambda r: r.utility_mean_avg)
        ax.annotate(
            f"{label}: {best.policy}",
            (best.coverage_avg, best.utility_mean_avg),
            fontsize=7,
        )

    ax.set_xlabel("Coverage (avg across seeds)")
    ax.set_ylabel("Utility mean (avg across seeds)")
    ax.set_title("Utility-Coverage Curves Across Models")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_results_markdown(
    path: Path,
    label: str,
    seed_dirs: list[Path],
    rows_sorted: list[PolicyStats],
    baseline: PolicyStats | None,
    best: PolicyStats,
    gain_vs_baseline: float,
    gain_vs_baseline_std: float,
    gain_vs_baseline_min: float,
    gain_vs_baseline_max: float,
    best_policy_counts: dict[str, int],
    best_domain_counts: dict[str, dict[str, int]],
    best_profile_counts: dict[str, dict[str, int]],
    plot_path: Path,
) -> None:
    safe_rows = [r for r in rows_sorted if r.severe_error_rate_avg <= 1e-12]
    best_safe = safe_rows[0] if safe_rows else None

    with path.open("w", encoding="utf-8") as f:
        f.write("# Results (Auto-generated from Real Outputs)\n\n")
        f.write(f"- Run label: `{label}`\n")
        f.write(f"- Number of seeds: {len(seed_dirs)}\n")
        if seed_dirs:
            f.write("- Seeds: " + ", ".join(d.name.replace("seed_", "") for d in seed_dirs) + "\n")
        f.write("\n")

        f.write("## Multi-Seed Main Result\n\n")
        f.write(
            f"- Best policy by mean utility: `{best.policy}` "
            f"(utility={best.utility_mean_avg:.4f} ± {best.utility_mean_std:.4f}, "
            f"coverage={best.coverage_avg:.4f} ± {best.coverage_std:.4f}, "
            f"severe_error_rate={best.severe_error_rate_avg:.4f} ± {best.severe_error_rate_std:.4f}).\n"
        )
        if baseline is not None:
            f.write(
                f"- Gain moyen vs `always_escalate`: {gain_vs_baseline:.4f} ± {gain_vs_baseline_std:.4f} "
                f"(min={gain_vs_baseline_min:.4f}, max={gain_vs_baseline_max:.4f}, "
                f"baseline utility={baseline.utility_mean_avg:.4f}).\n"
            )
        if best_safe is not None:
            f.write(
                f"- Best zero-severe policy: `{best_safe.policy}` "
                f"(utility={best_safe.utility_mean_avg:.4f}, coverage={best_safe.coverage_avg:.4f}).\n"
            )
        f.write("\n")

        f.write("## Inter-Seed Variability\n\n")
        f.write("- Seed-level winner counts (utility on each seed):\n")
        for policy, count in sorted(best_policy_counts.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"  - `{policy}`: {count}\n")
        f.write("\n")

        f.write("## Heterogeneity Across Slices\n\n")
        f.write("- Best policy counts by domain (aggregated over seeds):\n")
        for domain, policy_counts in best_domain_counts.items():
            terms = ", ".join(f"{k}:{v}" for k, v in sorted(policy_counts.items()))
            f.write(f"  - `{domain}` -> {terms}\n")
        f.write("- Best policy counts by profile (aggregated over seeds):\n")
        for profile, policy_counts in best_profile_counts.items():
            terms = ", ".join(f"{k}:{v}" for k, v in sorted(policy_counts.items()))
            f.write(f"  - `{profile}` -> {terms}\n")
        f.write("\n")

        f.write("## Utility-Coverage Curve\n\n")
        f.write(f"- Figure: `{plot_path}`\n")
        f.write(
            "- This curve is built from policy-level means across seeds "
            "(utility_mean_avg, coverage_avg).\n"
        )
        f.write("\n")

        f.write("## Policy Table (mean ± std over seeds)\n\n")
        for row in rows_sorted:
            f.write(
                f"- `{row.policy}`: utility={row.utility_mean_avg:.4f} ± {row.utility_mean_std:.4f}, "
                f"coverage={row.coverage_avg:.4f} ± {row.coverage_std:.4f}, "
                f"severe_error_rate={row.severe_error_rate_avg:.4f} ± {row.severe_error_rate_std:.4f}\n"
            )


def _ensure_matplotlib_cache() -> None:
    # Avoid non-writable $HOME in sandbox/Colab contexts.
    cache_dir = os.environ.get("MPLCONFIGDIR", "").strip()
    if cache_dir:
        return
    os.environ["MPLCONFIGDIR"] = str(Path(".mplconfig").resolve())


def _mean(values: list[float]) -> float:
    if len(values) == 0:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((x - m) * (x - m) for x in values) / (len(values) - 1)
    return variance ** 0.5
