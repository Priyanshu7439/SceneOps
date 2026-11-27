# SceneOps Verified Metrics

Date: 2026-11-27

## Dataset metrics

Source: data/processed/sceneops_narration_manifest.json

| Metric | Verified value | Status |
|---|---:|---|
| Total examples | 5,856 | Current checkout verified |
| Scenarios | 320 | Current checkout verified |
| Train examples/scenarios | 3,554 / 192 | Current checkout verified |
| Validation examples/scenarios | 1,121 / 64 | Current checkout verified |
| Test examples/scenarios | 1,181 / 64 | Current checkout verified |
| Scene examples | 2,801 | Current checkout verified |
| Temporal examples | 3,055 | Current checkout verified |
| Scenario leakage audit | Passed | Current checkout verified |

## Blind model metrics

Sources: artifacts/evaluation/base_blind/metrics.json and artifacts/evaluation/gguf_blind/metrics.json

| Metric | Base | Fine-tuned GGUF |
|---|---:|---:|
| Overall token F1 | 0.6235269268 | 0.5043649728 |
| Scene token F1 | 0.8181508992 | 0.5649590388 |
| Temporal token F1 | 0.4468247095 | 0.4493506512 |
| Completed examples | 1,181 | 1,181 |
| Failed examples | 0 | 0 |

Fine-tuned minus base: overall -11.9162 percentage points, scene -25.3192 percentage points, temporal +0.2526 percentage points.

Paired fine-tuned wins/losses/ties: overall 215/966/0, scene 0/562/0, temporal 215/404/0.

Evaluation settings: temperature 0, max tokens 256, identical dataset SHA, system/user-only prompts, assistant target excluded. These are lexical metrics, not semantic or clinical correctness.

## State-fidelity metrics

Source: artifacts/evaluation/semantic/state_fidelity_v2_report.json

| Metric | Value |
|---|---:|
| Examples | 1,181 |
| Overall state-fidelity pass rate | 1.0 |
| Patient identity accuracy | 1.0 |
| Timestamp accuracy | 1.0 |
| Observation coverage | 1.0 |
| Observation value accuracy | 1.0 |
| Scene examples | 562 |
| Temporal examples | 619 |

Scope: closed-grammar field-level evaluation against regenerated simulator observations. This does not establish clinical validity, clinical safety, or broad free-form semantic robustness.

## Current test metrics

Source: final local verification on 2026-11-27.

- Python: 532 passed, 0 failed, 0 skipped.
- Ruff src/tests: passed.
- Frontend lint: passed.
- Frontend production build: passed.
- Warnings: 5 existing dependency/deprecation/cache warnings.

## Verification labels

- Current checkout verified: inspected or executed in this environment.
- Preserved completed run: existing artifact inspected; not rerun.
- Historical evidence: supported by notebook/audit records; not locally reproducible.
- Live verified: real local provider/application request succeeded.
- Mock-tested: dependency-light or injected test, not live deployment.
- Unverified: no successful current live evidence.
