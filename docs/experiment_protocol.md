# Experimental Protocol (Paper)

## Research Questions
1. Does a domain- and user-profile-conditioned deferral policy improve overall utility?
2. Do policy rankings differ across user profiles?
3. Which domains remain effectively non-automatable under severe-risk constraints?

## Variables
- Unit: one LLM query instance with correctness label.
- Domain: `general`, `finance`, `medical`.
- User profile: `cautious_novice`, `balanced_user`, `expert_fast`.
- Confidence signals: `p_internal`, `p_verbal`, `agreement`.
- Safety label: `severe_if_wrong`.

## Compared Policies
- `always_escalate`
- `global_threshold`
- `domain_threshold`
- `profile_threshold`
- `domain_profile_threshold`
- `rtr_style_threshold_router`
- `learned_utility_router`

## Primary Metrics
- `utility_mean`
- `coverage`
- `escalation_rate`
- `accepted_accuracy`
- `severe_error_rate`

## Recommended Pipeline
1. Build a normalized CSV from real data (`collect-real` or `collect-local-hf`).
2. Use an official fixed train/test split when available (otherwise stratified split).
3. Calibrate confidence signals on train.
4. Train routing policies.
5. Evaluate on test + bootstrap confidence intervals.
6. Analyze policy rankings by slices.

## Robustness Checks (Strongly Recommended)
- If no official split: run at least 5 random seeds for stability analysis.
- Utility sensitivity analysis (profile cost parameters).
- Confidence signal ablations (`p_verbal`, `agreement`, etc.).
- Difficulty-stratified analysis (quartiles).

## Threats to Validity
- Synthetic user profiles: useful for heterogeneity testing, but should be validated with real user studies.
- `severe_if_wrong` labels depend on annotation quality.
- Potential distribution shifts across domains.
