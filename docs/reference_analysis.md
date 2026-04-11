# Reference Analysis and Experimental Implications

## 1) Calibration and confidence (Geng et al., NAACL 2024)
- Key message: LLM confidence signals are imperfect and sensitive to prompting and estimation method.
- Design implication: do not use `p_internal` alone; combine multiple signals (`p_internal`, `p_verbal`, `agreement`) and calibrate.

## 2) Non-universal thresholds (Kumar et al., OCIT 2025)
- Key message: the best threshold varies by model and category.
- Design implication: compare `global`, `domain`, `profile`, and `domain×profile` thresholds.

## 3) Trust-or-Escalate (Jung et al., 2024)
- Key message: selective trust must be explicitly evaluated with risk-aware criteria.
- Design implication: report `coverage`, `accepted_accuracy`, and `severe_error_rate`, not only raw accuracy.

## 4) Abstention in LLMs (Wen et al., TACL 2025)
- Key message: context-aware abstention is central, not peripheral.
- Design implication: formalize abstention as a policy conditioned on user and task.

## 5) Human-Centered Deferred Inference (Lemmer et al., IUI 2023)
- Key message: user interaction patterns affect optimal deferral behavior.
- Design implication: model user profiles with different error/escalation costs.

## 6) LLM Performance Predictors (Bachar et al., 2026)
- Key message: a meta-model can predict when to escalate.
- Design implication: include a learned router baseline (`learned_utility_router`).

## 7) SCOPE / risk control (Badshah et al., 2026)
- Key message: risk among accepted cases should be explicitly controlled.
- Design implication: analyze `severe_error_rate` and calibrate policies to risk-sensitive utilities.

## 8) Route to Reason (Pan et al., 2025)
- Key message: adaptive routing often outperforms fixed rules.
- Design implication: transfer this logic to `LLM-autonomous` vs `human` routing.

## 9) FinanceBench (Islam et al., 2023)
- Key message: finance QA remains brittle for safe automation.
- Design implication: expect low safe coverage under conservative constraints in finance.

## 10) Long-form calibration (Huang et al., 2024)
- Key message: calibration is harder for open-ended generation than for narrow factoid tasks.
- Design implication: in future extensions, move beyond binary `correctness` to graded outcomes.

## Testable Hypotheses
1. `domain×profile` policies improve utility over `global_threshold`.
2. Policy rankings vary across user profiles.
3. Finance/medical domains have lower safe coverage than general reasoning.
4. Learned routing benefits risk-sensitive profiles the most.
