# SceneOps Repository Audit

Date: 2026-11-27

## Git state

- Branch: main.
- Remote: https://github.com/Priyanshu7439/SceneOps.git.
- Final local HEAD: 31309b3d334fae232b2bb4e8ae2d7403e09d7d6e.
- origin/main verified at the same commit after push.
- Final tracked working tree: clean.
- Preserved untracked files remain intentionally untouched.

## Release commits

1. 9380a8e docs: finalize SceneOps documentation and verified project release
2. 31309b3 Merge origin/main release documentation updates

The second commit preserved three remote README-only commits containing user-provided screenshot links while retaining the audited release documentation.

## Repository structure

- src/sceneops/simulator: deterministic simulation rules.
- src/sceneops/agents: typed contracts, narration, dialogue, orchestration, safety.
- src/sceneops/application: application service.
- src/sceneops/api: FastAPI API, schemas, session management.
- src/sceneops/serving: OpenAI-compatible client and Ray Serve adapter.
- src/sceneops/training: dataset generation and validation.
- src/sceneops/tracking: optional W&B wrapper.
- frontend: React, TypeScript, Vite dashboard.
- tests: simulator, agents, API, application, evaluation, serving, tracking, and training.
- docs: professional documentation and completion audits.
- artifacts/evaluation: selected metrics, manifests, summaries, and state-fidelity report.

## Release contents

The release commit includes application source, tests, frontend source/configuration, Axolotl reference configuration, serving/tracking helpers, evaluation metadata, README, documentation, contributing guidance, changelog, and screenshot instructions.

## Excluded but preserved

The following remain local and were not deleted:

- models/gguf/sceneops-q4_k_m.gguf and other large model artifacts.
- Raw blind predictions and state-fidelity prediction JSONL.
- Target-leaking artifacts/evaluation/gguf_final.
- Recovery patches, audit dumps, ZIP archives, and temporary reconstruction files.
- Local notebook audit exports.
- Any local credentials or environment files.

Large weights, checkpoints, private data, raw generated predictions, and temporary forensic material were excluded to avoid bloating or leaking the GitHub release.

## Security audit

- No staged API keys, W&B tokens, private-key blocks, or bearer credentials detected.
- .env is ignored; .env.example contains configuration placeholders only.
- Frontend production code contains no provider secrets.
- CORS is explicit/configurable rather than a production wildcard.
- Provider configuration is separated from simulator business logic.
- No public model-weight deployment was performed.

## Quality gates

- git diff --cached --check: passed before release commit.
- Full Python suite: 532 passed.
- Ruff src/tests: passed.
- Frontend lint: passed.
- Frontend build: passed.
- Real local llama.cpp scene and temporal inference: passed.
- Live Ray Serve inference: not verified.
- Live vLLM server inference: not verified.
- Remote W&B run: not verified.
- Axolotl training: not executed.

## Related reports

- docs/completion/final_project_status.md
- docs/completion/model_comparison.md
- docs/completion/semantic_evaluation.md
- docs/completion/training_audit.md
- docs/completion/reproducibility.md
- docs/completion/limitations_and_safety.md
- docs/completion/artifact_inventory.md
