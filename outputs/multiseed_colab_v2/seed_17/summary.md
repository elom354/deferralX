# DeferralX Experiment Summary

- Total samples: 600
- Train samples: 425
- Test samples: 175

## Best Overall Policy

- learned_utility_router: utility=-0.3817, coverage=0.1314, accepted_accuracy=0.5217, severe_error_rate=0.0000

## Best Policy by Domain

- finance: always_escalate (utility=-0.4071)
- general: learned_utility_router (utility=-0.3547)
- medical: always_escalate (utility=-0.4000)

## Best Policy by User Profile

- balanced_user: always_escalate (utility=-0.4000)
- cautious_novice: domain_profile_threshold (utility=-0.5458)
- expert_fast: learned_utility_router (utility=-0.1414)

## Heterogeneity Check

- Overall best policy: learned_utility_router
- Distinct best policies across domains: ['always_escalate', 'learned_utility_router']
- Distinct best policies across profiles: ['always_escalate', 'domain_profile_threshold', 'learned_utility_router']
