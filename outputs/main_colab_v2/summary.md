# DeferralX Experiment Summary

- Total samples: 600
- Train samples: 425
- Test samples: 175

## Best Overall Policy

- always_escalate: utility=-0.4011, coverage=0.0000, accepted_accuracy=NA, severe_error_rate=0.0000

## Best Policy by Domain

- finance: always_escalate (utility=-0.4071)
- general: always_escalate (utility=-0.4000)
- medical: always_escalate (utility=-0.4000)

## Best Policy by User Profile

- balanced_user: always_escalate (utility=-0.4000)
- cautious_novice: always_escalate (utility=-0.6000)
- expert_fast: domain_threshold (utility=-0.1931)

## Heterogeneity Check

- Overall best policy: always_escalate
- Distinct best policies across domains: ['always_escalate']
- Distinct best policies across profiles: ['always_escalate', 'domain_threshold']
