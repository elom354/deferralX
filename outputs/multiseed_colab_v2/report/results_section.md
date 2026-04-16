# Results (Auto-generated from Real Outputs)

- Run label: `qwen2.5-0.5b`
- Number of seeds: 5
- Seeds: 11, 17, 23, 31, 7

## Multi-Seed Main Result

- Best policy by mean utility: `learned_utility_router` (utility=-0.3483 ± 0.0315, coverage=0.1257 ± 0.0107, severe_error_rate=0.0000 ± 0.0000).
- Gain moyen vs `always_escalate`: 0.0528 ± 0.0315 (min=0.0194, max=0.0869, baseline utility=-0.4011).
- Best zero-severe policy: `learned_utility_router` (utility=-0.3483, coverage=0.1257).

## Inter-Seed Variability

- Seed-level winner counts (utility on each seed):
  - `learned_utility_router`: 3
  - `domain_profile_threshold`: 2

## Heterogeneity Across Slices

- Best policy counts by domain (aggregated over seeds):
  - `finance` -> always_escalate:4, domain_profile_threshold:3, domain_threshold:4, global_threshold:4, learned_utility_router:4, profile_threshold:4, rtr_style_threshold_router:3
  - `general` -> domain_profile_threshold:3, learned_utility_router:2
  - `medical` -> always_escalate:5, domain_profile_threshold:4, domain_threshold:5, global_threshold:5, learned_utility_router:5, profile_threshold:5, rtr_style_threshold_router:5
- Best policy counts by profile (aggregated over seeds):
  - `balanced_user` -> always_escalate:2, domain_profile_threshold:3, domain_threshold:1, global_threshold:2, learned_utility_router:2, profile_threshold:2, rtr_style_threshold_router:2
  - `cautious_novice` -> always_escalate:4, domain_profile_threshold:5, global_threshold:4, learned_utility_router:4, profile_threshold:4, rtr_style_threshold_router:4
  - `expert_fast` -> domain_profile_threshold:2, learned_utility_router:3

## Utility-Coverage Curve

- Figure: `outputs/multiseed_colab_v2/report/utility_coverage_curve.png`
- This curve is built from policy-level means across seeds (utility_mean_avg, coverage_avg).

## Policy Table (mean ± std over seeds)

- `learned_utility_router`: utility=-0.3483 ± 0.0315, coverage=0.1257 ± 0.0107, severe_error_rate=0.0000 ± 0.0000
- `always_escalate`: utility=-0.4011 ± 0.0000, coverage=0.0000 ± 0.0000, severe_error_rate=0.0000 ± 0.0000
- `global_threshold`: utility=-0.4011 ± 0.0000, coverage=0.0000 ± 0.0000, severe_error_rate=0.0000 ± 0.0000
- `profile_threshold`: utility=-0.4011 ± 0.0000, coverage=0.0000 ± 0.0000, severe_error_rate=0.0000 ± 0.0000
- `rtr_style_threshold_router`: utility=-0.4141 ± 0.0289, coverage=0.0011 ± 0.0026, severe_error_rate=0.0011 ± 0.0026
- `domain_profile_threshold`: utility=-0.4618 ± 0.1540, coverage=0.1829 ± 0.0277, severe_error_rate=0.0091 ± 0.0104
- `domain_threshold`: utility=-0.5032 ± 0.1528, coverage=0.1131 ± 0.0930, severe_error_rate=0.0034 ± 0.0077
