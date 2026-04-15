# DeferralX Experiment Summary

- Total samples: 1500
- Train samples: 1054
- Test samples: 446

## Best Overall Policy

- learned_utility_router: utility=-0.3404, coverage=0.1839, accepted_accuracy=0.7317, severe_error_rate=0.0000

## Best Policy by Domain

- finance: profile_threshold (utility=-0.4200)
- general: domain_profile_threshold (utility=-0.2375)
- medical: always_escalate (utility=-0.4267)

## Best Policy by User Profile

- balanced_user: learned_utility_router (utility=-0.3536)
- cautious_novice: profile_threshold (utility=-0.5214)
- expert_fast: learned_utility_router (utility=0.1010)

## Heterogeneity Check

- Overall best policy: learned_utility_router
- Distinct best policies across domains: ['always_escalate', 'domain_profile_threshold', 'profile_threshold']
- Distinct best policies across profiles: ['learned_utility_router', 'profile_threshold']
