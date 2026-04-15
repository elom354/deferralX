# DeferralX Experiment Summary

- Total samples: 300
- Train samples: 240
- Test samples: 60

## Best Overall Policy

- domain_threshold: utility=-0.3350, coverage=0.1167, accepted_accuracy=0.8571, severe_error_rate=0.0000

## Best Policy by Domain

- finance: always_escalate (utility=-0.3889)
- general: global_threshold (utility=-0.2840)
- medical: domain_threshold (utility=-0.3529)

## Best Policy by User Profile

- balanced_user: domain_threshold (utility=-0.3160)
- cautious_novice: profile_threshold (utility=-0.5273)
- expert_fast: global_threshold (utility=0.0769)

## Heterogeneity Check

- Overall best policy: domain_threshold
- Distinct best policies across domains: ['always_escalate', 'domain_threshold', 'global_threshold']
- Distinct best policies across profiles: ['domain_threshold', 'global_threshold', 'profile_threshold']
