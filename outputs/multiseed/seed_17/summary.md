# DeferralX Experiment Summary

- Total samples: 1500
- Train samples: 1054
- Test samples: 446

## Best Overall Policy

- domain_threshold: utility=-0.3397, coverage=0.0919, accepted_accuracy=0.8780, severe_error_rate=0.0000

## Best Policy by Domain

- finance: domain_threshold (utility=-0.3966)
- general: domain_threshold (utility=-0.2630)
- medical: global_threshold (utility=-0.4167)

## Best Policy by User Profile

- balanced_user: global_threshold (utility=-0.2917)
- cautious_novice: domain_threshold (utility=-0.5143)
- expert_fast: learned_utility_router (utility=-0.0557)

## Heterogeneity Check

- Overall best policy: domain_threshold
- Distinct best policies across domains: ['domain_threshold', 'global_threshold']
- Distinct best policies across profiles: ['domain_threshold', 'global_threshold', 'learned_utility_router']
