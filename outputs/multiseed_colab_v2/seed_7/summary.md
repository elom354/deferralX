# DeferralX Experiment Summary

- Total samples: 600
- Train samples: 425
- Test samples: 175

## Best Overall Policy

- learned_utility_router: utility=-0.3143, coverage=0.1143, accepted_accuracy=0.8000, severe_error_rate=0.0000

## Best Policy by Domain

- finance: domain_threshold (utility=-0.3571)
- general: learned_utility_router (utility=-0.1973)
- medical: always_escalate (utility=-0.4000)

## Best Policy by User Profile

- balanced_user: always_escalate (utility=-0.4000)
- cautious_novice: always_escalate (utility=-0.6000)
- expert_fast: learned_utility_router (utility=0.0621)

## Heterogeneity Check

- Overall best policy: learned_utility_router
- Distinct best policies across domains: ['always_escalate', 'domain_threshold', 'learned_utility_router']
- Distinct best policies across profiles: ['always_escalate', 'learned_utility_router']
