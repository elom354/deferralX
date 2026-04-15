# DeferralX Experiment Summary

- Total samples: 300
- Train samples: 214
- Test samples: 86

## Best Overall Policy

- learned_utility_router: utility=-0.3605, coverage=0.0814, accepted_accuracy=0.7143, severe_error_rate=0.0000

## Best Policy by Domain

- finance: always_escalate (utility=-0.3900)
- general: learned_utility_router (utility=-0.3300)
- medical: always_escalate (utility=-0.3846)

## Best Policy by User Profile

- balanced_user: domain_threshold (utility=-0.3079)
- cautious_novice: profile_threshold (utility=-0.5846)
- expert_fast: learned_utility_router (utility=-0.0727)

## Heterogeneity Check

- Overall best policy: learned_utility_router
- Distinct best policies across domains: ['always_escalate', 'learned_utility_router']
- Distinct best policies across profiles: ['domain_threshold', 'learned_utility_router', 'profile_threshold']
