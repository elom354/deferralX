# Initial Results (Synthetic, Non-Publishable)

Configuration:
- Dataset: 1500 synthetic samples
- Split: stratified 70/30 by `domain x profile`
- Seeds: 7, 11, 17

## Mean overall result (3 seeds)
Source: `outputs/multiseed/multiseed_summary.csv`

- learned_utility_router: utility_mean_avg = -0.3711, coverage_avg = 0.1898, severe_error_rate_avg = 0.0000
- global_threshold: utility_mean_avg = -0.4105, coverage_avg = 0.0987, severe_error_rate_avg = 0.0022
- profile_threshold: utility_mean_avg = -0.4280, coverage_avg = 0.1584, severe_error_rate_avg = 0.0060
- always_escalate: utility_mean_avg = -0.4318, coverage_avg = 0.0000, severe_error_rate_avg = 0.0000
- domain_profile_threshold: utility_mean_avg = -0.5042, coverage_avg = 0.1719, severe_error_rate_avg = 0.0075
- domain_threshold: utility_mean_avg = -0.5055, coverage_avg = 0.0874, severe_error_rate_avg = 0.0052

## Interpretation
1. The learned router leads on average utility in these synthetic runs while keeping severe-error rate at zero.
2. Policy rankings vary by domain/profile (see per-seed `summary.md`).
3. Safe coverage remains limited in high-risk settings, consistent with a non-one-size-fits-all framing.

## Caution
These numbers are from synthetic data and are **not publishable evidence**. Validate on real benchmarks (e.g., FinanceBench and medical/general QA sets).
