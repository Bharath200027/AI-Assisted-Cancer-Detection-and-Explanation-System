# Security & Safety Guidance (Research)

## Core principles
- **Not a diagnosis:** Always display disclaimers.
- **Human-in-the-loop:** Use confidence gating and prioritize uncertain cases for review.
- **Auditability:** Log model version, plan, tool trace, and outputs.
- **Data governance:** Avoid storing PHI; encrypt at rest; restrict access.

## Deployment safety checklist
- [ ] Validate on external test set and institution-specific data
- [ ] Calibrate probabilities, define abstention thresholds
- [ ] Implement access controls and authentication
- [ ] Monitor drift and performance over time
- [ ] Establish clinical workflow and escalation policies
