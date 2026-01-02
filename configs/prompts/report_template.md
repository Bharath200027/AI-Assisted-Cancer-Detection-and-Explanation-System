# AI-Assisted Blood Cancer Screening Report (Research)

## Input
- File: {{image_filename}}

## Model Output
- Predicted class: **{{predicted_label}}**
- Confidence: **{{confidence}}**
- Class probabilities:
{{prob_table}}

## Visual Evidence (Explainability)
- Method: {{explain_method}}
- Heatmap file: {{heatmap_path}}
- Summary: {{explain_summary}}

## Retrieved Evidence (Literature / Guidelines Snippets)
{{evidence_block}}

## Interpretation (Cautious)
{{interpretation}}

## Limitations
{{limitations}}

---

## Two-stage Subtype Result (if screening = leukemia)
- Subtype: **{{predicted_subtype}}**
- Subtype confidence: **{{subtype_confidence}}**
- Subtype probabilities:
{{subtype_prob_table}}
- Subtype ensemble disagreement: {{subtype_disagreement}}

## Stage-1 Screening Snapshot (audit)
- Stage-1 predicted label: {{stage1_predicted_label}}
- Stage-1 confidence: {{stage1_confidence}}
- Stage-1 model: {{stage1_model}}
- Stage-1 checkpoint: {{stage1_checkpoint}}
