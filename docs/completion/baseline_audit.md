# Baseline audit

Audit date: 2026-11-27. Branch: `main`. HEAD: `f8732ab` (`docs: add comprehensive SceneOps project README`). The working tree already contained extensive uncommitted user changes; none were reverted.

## Repository evidence

- Simulator: `src/sceneops/simulator/` (patients, observations, deterioration, clock, triage, resources, transport, scenario generation, and engine).
- Agent boundary: `src/sceneops/agents/` (disclosures, scene narration, temporal narration, patient dialogue, orchestration, and rule-based safety review).
- Application/API: `src/sceneops/application/` and `src/sceneops/api/`.
- Training/data: `src/sceneops/training/`, `data/processed/`, and `notebooks/`.
- Evaluation: `scripts/evaluate_gguf_narration.py`, `scripts/evaluate_sceneops_state_fidelity_v2.py`, `src/sceneops/evaluation/`, and preserved `artifacts/evaluation/`.
- Model artifact: `models/gguf/sceneops-q4_k_m.gguf` (986,048,160 bytes).

## Dataset verification

The four processed narration files contain 5,856 master, 3,554 train, 1,121 validation, and 1,181 test records. The manifest reports 320 scenarios and passed scenario-level splitting. SHA-256 hashes match the manifest for master, train, validation, and test.

## Baseline tests

The first run without environment adjustment produced 499 passed and 19 setup errors because the host denied pytest's default temp directory. Re-running with `TEMP` and `TMP` set to a writable workspace directory produced **518 passed, 0 failed, 0 skipped**, with 5 dependency/deprecation warnings. No code was changed to bypass a test assertion.

## Existing modifications

`git status --short` showed pre-existing edits in agent, API, application, simulator, and test files, plus untracked evaluation, frontend, model, notebook, and state-fidelity artifacts. These remain preserved.
