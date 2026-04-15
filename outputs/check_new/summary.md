# DeferralX Experiment Summary

- Total samples: 300
- Train samples: 214
- Test samples: 86

## Best Overall Policy

- rtr_style_threshold_router: utility=-0.3884, coverage=0.1163, accepted_accuracy=0.6000, severe_error_rate=0.0000

## Best Policy by Domain

- finance: always_escalate (utility=-0.3900)
- general: rtr_style_threshold_router (utility=-0.3900)
- medical: always_escalate (utility=-0.3846)

## Best Policy by User Profile

- balanced_user: always_escalate (utility=-0.4000)
- cautious_novice: global_threshold (utility=-0.4615)
- expert_fast: rtr_style_threshold_router (utility=-0.1273)

## Heterogeneity Check

- Overall best policy: rtr_style_threshold_router
- Distinct best policies across domains: ['always_escalate', 'rtr_style_threshold_router']
- Distinct best policies across profiles: ['always_escalate', 'global_threshold', 'rtr_style_threshold_router']
