# DeferralX Experiment Summary

- Total samples: 1500
- Train samples: 1054
- Test samples: 446

## Best Overall Policy

- learned_utility_router: utility=-0.4126, coverage=0.2040, accepted_accuracy=0.5824, severe_error_rate=0.0000

## Best Policy by Domain

- finance: always_escalate (utility=-0.4373)
- general: domain_profile_threshold (utility=-0.3188)
- medical: profile_threshold (utility=-0.3950)

## Best Policy by User Profile

- balanced_user: always_escalate (utility=-0.4000)
- cautious_novice: profile_threshold (utility=-0.5333)
- expert_fast: domain_profile_threshold (utility=-0.0788)

## Heterogeneity Check

- Overall best policy: learned_utility_router
- Distinct best policies across domains: ['always_escalate', 'domain_profile_threshold', 'profile_threshold']
- Distinct best policies across profiles: ['always_escalate', 'domain_profile_threshold', 'profile_threshold']
