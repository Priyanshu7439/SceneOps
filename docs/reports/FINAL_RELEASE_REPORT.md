# SceneOps Final Release Report

Date: 2026-11-27

## Release identity

- Repository: https://github.com/Priyanshu7439/SceneOps
- Branch: main
- Final pushed commit: 31309b3d334fae232b2bb4e8ae2d7403e09d7d6e
- Release commit: 9380a8e docs: finalize SceneOps documentation and verified project release
- Merge commit: 31309b3 Merge origin/main release documentation updates
- Push confirmation: origin/main resolved to 31309b3d334fae232b2bb4e8ae2d7403e09d7d6e

## Project

SceneOps is an educational synthetic mass-casualty incident simulator. Python is the sole authority for patient state, deterioration, clock progression, observations, triage, ambulances, transport, survival, outcomes, events, and scoring. The LLM generates scene and temporal narration only from authorized disclosures. Rule-based review rejects unsupported or unsafe output.

## Verified implementation

- Deterministic simulator with 20-patient scenario support.
- FastAPI backend and React/TypeScript/Vite frontend.
- Scene narration and temporal narration agents.
- Typed agent contracts, orchestration, disclosure boundaries, and rule-based safety review.
- OpenAI-compatible model client with actionable provider failures.
- Real llama.cpp GGUF inference verified for scene and temporal narration.
- Optional Ray Serve adapter, W&B tracker, and Axolotl reference configuration.

## Dataset

Authoritative manifest: data/processed/sceneops_narration_manifest.json.

- 5,856 unique examples.
- 320 scenarios.
- Train: 3,554 examples across 192 scenarios.
- Validation: 1,121 examples across 64 scenarios.
- Test: 1,181 examples across 64 scenarios.
- Scene examples: 2,801.
- Temporal examples: 3,055.
- Scenario-level leakage audit: passed.

## Training and artifacts

Historical training used Qwen/Qwen2.5-1.5B-Instruct with TRL SFTTrainer, PEFT LoRA/QLoRA, bitsandbytes 4-bit NF4, rank 16, alpha 32, dropout 0.05, q/k/v/o/gate/up/down target modules, 3 epochs, 669 optimization steps, and resume from checkpoint-223.

This is historical evidence from the notebook audit. The original checkpoint and complete GPU environment are not present locally. No retraining was performed.

Historical merged Hugging Face, AWQ, and GGUF Q4_K_M exports are documented. The GGUF is intended for llama.cpp-compatible serving; vLLM GGUF compatibility was not assumed.

## Blind evaluation

Valid artifacts are artifacts/evaluation/base_blind and artifacts/evaluation/gguf_blind. Both evaluated 1,181 examples with identical dataset SHA, temperature 0, max tokens 256, and system/user-only prompts with assistant targets excluded.

| Metric | Base | Fine-tuned GGUF | Delta |
|---|---:|---:|---:|
| Overall token F1 | 0.6235269 | 0.5043650 | -11.9162 pp |
| Scene token F1 | 0.8181509 | 0.5649590 | -25.3192 pp |
| Temporal token F1 | 0.4468247 | 0.4493507 | +0.2526 pp |

Fine-tuned wins/losses/ties: 215/966/0 overall, 0/562/0 scene, and 215/404/0 temporal. Fine-tuning did not improve preserved lexical F1. The target-leaking artifacts/evaluation/gguf_final output is excluded from conclusions.

## State fidelity

The independent report artifacts/evaluation/semantic/state_fidelity_v2_report.json records 1,181/1,181 strict-template checks passing for identity, timestamps, observation coverage, and values. It is a closed-grammar field-level evaluator, not broad semantic validation or clinical validation.

The detailed audit records seven temporal follow-command attribution errors, visible hemorrhage explicitly correct in 606/619 applicable cases, and 16 capillary-refill omissions where available.

## Serving and assignment status

- TRL/PEFT: historically verified.
- AWQ/GGUF: historical conversion evidence preserved.
- llama.cpp: real local scene and temporal inference verified.
- vLLM: client and mocked HTTP behavior tested; live vLLM server inference unverified.
- Ray Serve: implementation and mocked health/error/timeout tests verified; live model inference unverified.
- W&B: disabled/offline/mock-safe tracker verified; remote logging unverified.
- Axolotl: configuration structurally validated; no Axolotl training run executed.

## Application verification

- Backend health and session lifecycle verified.
- Frontend proxy health and session creation verified.
- Scene narration verified through the real provider.
- Temporal narration verified across simulation seconds 0, 60, and 120.
- Provider failure returned HTTP 503 without simulator state mutation.
- Frontend production build passed.
- Browser visual automation was unavailable; HTTP-level frontend integration was verified.

## Tests

- Python: 532 passed, 0 failed, 0 skipped.
- Ruff for src and tests: passed.
- Frontend lint: passed.
- Frontend production build: passed.
- Existing warnings: 5 dependency/deprecation/cache warnings.

## Repository and safety

No credentials were found in staged release text. Local .env files, GGUF weights, checkpoints, node modules, raw prediction JSONL, recovery bundles, and temporary audit files were excluded from the release commit and preserved locally. No license, DOI, clinical endorsement, production benchmark, or remote W&B run was invented.

## Final limitations

SceneOps is suitable for an educational software demonstration and documented engineering portfolio. It must not be presented as clinically validated, medically reliable, production-deployed, or evidence that fine-tuning improved the base model.
