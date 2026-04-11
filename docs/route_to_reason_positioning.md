# Positioning vs Route-To-Reason

Reference: https://github.com/goodmanpzh/Route-To-Reason

## What Route-To-Reason does
- Adaptive routing across `(LLM, reasoning strategy)` pairs.
- Objective: performance-cost tradeoff for generation.
- Mechanism: predict performance/cost and route to the best option.

## What DeferralX does
- Adaptive routing between `autonomous LLM` and `human oversight`.
- Objective: utility under risk (severe error cost, escalation cost, user preference).
- Mechanism: threshold policies + learned routers conditioned on `domain x user profile`.

## Intended empirical contribution
1. Show that the best policy is not universal.
2. Quantify how user profile shifts the optimal operating point.
3. Identify domains that remain near-non-automatable under strict safety constraints.
4. Compare an `rtr_style_threshold_router` baseline against a utility-driven human-deferral router.
