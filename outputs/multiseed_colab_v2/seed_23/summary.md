# DeferralX Experiment Summary

- Total samples: 600
- Train samples: 425
- Test samples: 175

## Best Overall Policy

- domain_profile_threshold: utility=-0.3469, coverage=0.1600, accepted_accuracy=0.6429, severe_error_rate=0.0000

## Best Policy by Domain

- finance: always_escalate (utility=-0.4071)
- general: domain_profile_threshold (utility=-0.2733)
- medical: always_escalate (utility=-0.4000)

## Best Policy by User Profile

- balanced_user: domain_profile_threshold (utility=-0.3638)
- cautious_novice: always_escalate (utility=-0.6000)
- expert_fast: domain_profile_threshold (utility=-0.0724)

## Heterogeneity Check

- Overall best policy: domain_profile_threshold
- Distinct best policies across domains: ['always_escalate', 'domain_profile_threshold']
- Distinct best policies across profiles: ['always_escalate', 'domain_profile_threshold']
