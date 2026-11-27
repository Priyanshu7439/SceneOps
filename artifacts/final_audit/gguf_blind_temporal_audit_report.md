# SceneOps ? Blind GGUF Evaluation & Temporal Audit

## 1. Evaluation setup
- Base model: Qwen2.5-1.5B-Instruct
- Fine-tuning: QLoRA / LoRA
- Quantization: GGUF Q4_K_M
- Blind test examples: 1,181
- Completed predictions: 1,181
- Final failed predictions: 0

Historical note: Results in artifacts/evaluation/gguf_final are excluded because the earlier evaluator leaked assistant targets.

## 2. Blind lexical evaluation

| Metric | Overall |
|---|---:|
| Exact match | 0.0000 |
| Token precision | 0.5908 |
| Token recall | 0.4703 |
| Token F1 | 0.5044 |
| Sequence similarity | 0.2793 |

| Task | Examples | Precision | Recall | F1 | Sequence similarity |
|---|---:|---:|---:|---:|---:|
| Scene | 562 | 0.5273 | 0.6085 | 0.5650 | 0.4273 |
| Temporal | 619 | 0.6485 | 0.3448 | 0.4494 | 0.1449 |

These are lexical metrics, not semantic or clinical accuracy.

## 3. Temporal transition checks

| Field | Changed cases | Exact matches | Errors |
|---|---:|---:|---:|
| Respiratory rate | 230 | 230 | 0 |
| Radial pulse | 63 | 63 | 0 |
| Mental status | 71 | 71 | 0 |
| Ambulatory status | 55 | 55 | 0 |
| Follows commands | 71 | 64 | 7 |

Seven follows-commands errors incorrectly projected the follow-up state onto both observation points.

Affected prediction indices: 233, 323, 828, 842, 972, 1140, 1166.

## 4. Capillary refill audit
- Both observation values available: 601
- Ground-truth transitions: 0
- Exact values explicitly preserved: 585
- Omissions / unparsed: 16
- Explicit incorrect numerical values: 0
- Both observation values missing: 18
- Numerical hallucinations in those 18 cases: 0

Among the 16 omissions, 13 used generic unchanged-observation wording and 3 omitted capillary refill entirely.

## 5. Visible hemorrhage audit

| Classification | Cases |
|---|---:|
| Correct negative | 504 |
| Correct positive | 102 |
| Contradictions | 0 |
| Omissions | 13 |
| Total | 619 |

Explicitly correct status: 606/619 (97.90%).

The 13 omissions are missing explicit status statements, not confirmed contradictory claims.

## 6. Ground-truth transition inventory

| Field | Changed | Unchanged | Missing |
|---|---:|---:|---:|
| Respiratory rate | 230 | 375 | 14 |
| Radial pulse | 63 | 556 | 0 |
| Capillary refill | 0 | 601 | 18 |
| Mental status | 71 | 548 | 0 |
| Ambulatory status | 55 | 564 | 0 |
| Visible hemorrhage | 0 | 619 | 0 |
| Follows commands | 71 | 548 | 0 |

## 7. Interpretation and limitations
- The overall blind lexical F1 is 0.5044; temporal F1 is 0.4494.
- Seven follows-commands temporal attribution errors were confirmed.
- Capillary refill and visible hemorrhage had omissions.
- These are field-specific findings; counts may overlap across predictions and must not be added as unique faulty examples.
- This is not a complete semantic evaluation of every generated sentence or field.
- No clinical accuracy or clinical safety claim is established by these results.
- The simulator remains the ground truth; the LLM generates language only.

## 8. Metric policy
Do not manually alter the measured F1. Any model, prompt, decoding, or evaluator change requires a fresh blind evaluation. Preserve this evaluation as the baseline.

## 9. Source artifacts
- artifacts/evaluation/gguf_blind/metrics.json
- artifacts/evaluation/gguf_blind/evaluation_summary.md
- artifacts/evaluation/gguf_blind/predictions.jsonl
- artifacts/evaluation/gguf_blind/evaluation_manifest.json
- artifacts/final_audit/sceneops_test_ground_truth_mapping.jsonl
