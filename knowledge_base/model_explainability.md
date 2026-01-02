# Model & Explainability Notes

- Classification models can be overconfident; use probability calibration and thresholds.
- Grad-CAM highlights *correlated* regions, not guaranteed causal evidence.
- Prefer human review for borderline cases and out-of-distribution images.

Recommended operational practice:
- Track data drift and periodically re-validate.
- Maintain audit logs of model version, preprocessing, and prompts.
