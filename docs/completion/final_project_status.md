# Final project status

## Summary

SceneOps/TriageCommand is a deterministic mass-casualty training simulator with a narration boundary. The simulator owns patient state, observations, deterioration, triage, ambulance/resource allocation, transport, scoring, and outcomes. Scene and temporal narration agents receive only validated disclosures, and a rule-based safety reviewer rejects unsupported or clinical-decision content.

## Verified

- Dataset manifest and JSONL counts/hash integrity: 5,856 total examples, 320 scenarios, 3,554/1,121/1,181 train/validation/test; scenario-level split status passed.
- Full local suite: 518 passed, 0 failed, 0 skipped, 5 warnings with a writable temp override.
- Final full local suite after completion changes: 530 passed, 0 failed, 0 skipped, 4 warnings; Ruff clean.
- Frontend production build passes with `npm run build` (TypeScript and Vite).
- Backend/frontend readiness audit completed; see `deployment_readiness.md`.
- Actual local HTTP integration completed with a deterministic mock provider; browser automation was blocked by an unavailable browser host.
- Real model-provider integration is verified through the live llama.cpp OpenAI-compatible endpoint at `127.0.0.1:8081`: direct inference, scene narration, and temporal narration all succeeded.
- Temporal regression repaired: real frontend-proxy verification now returns approved temporal narration across seconds 0, 60, and 120 without changing simulator state.
- Fixed and verified the Vite-to-backend proxy port mismatch; proxied `/health` returned 200 and `/sessions` returned 201.
- State-fidelity suite: 10 adversarial tests passed; additional boundary tests cover model failure and state immutability; held-out strict-template report: 1,181/1,181 pass.
- Valid blind comparison completed on the same 1,181 examples for base and GGUF endpoints.
- GGUF exists locally at `models/gguf/sceneops-q4_k_m.gguf`, 986,048,160 bytes.
- Existing simulator/API/agent work and patient-dialogue datasets were preserved.
- The 503 was traced to stale port-8000 and placeholder-model defaults; simulator authority was unchanged.

## Model result

The fine-tuned model is not better than base by lexical F1: overall 0.5043650 versus 0.6235269; scene 0.5649590 versus 0.8181509; temporal 0.4493507 versus 0.4468247. The result is a regression, not a success claim. State-fidelity results are strong for the closed grammar but do not cancel the regression or prove clinical safety.

## Assignment technology status

- TRL/PEFT: VERIFIED COMPLETE historically; not installed or live-rerun locally.
- AWQ/GGUF: VERIFIED COMPLETE historically; local GGUF exists and historical smoke evidence is preserved; cryptographic provenance is unavailable.
- Ray Serve: IMPLEMENTED BUT NOT LIVE-VERIFIED via `src/sceneops/serving/ray_serve_app.py`; mocked timeout/error/health tests pass.
- vLLM-compatible client: VERIFIED against the live llama.cpp OpenAI-compatible API; actual vLLM server/runtime remains NOT LIVE-VERIFIED, and GGUF is not forced through vLLM.
- W&B: IMPLEMENTED BUT NOT LIVE-VERIFIED; disabled/offline/mock-safe tracker added, no remote upload.
- Axolotl: IMPLEMENTED BUT NOT LIVE-VERIFIED as a configuration path; structural validation passes, actual training was not executed. Historical training was TRL/PEFT.

## Reproduction and blockers

See `reproducibility.md` and `deployment_readiness.md` for commands. Live serving requires optional packages, a compatible model runtime, and (for remote W&B) credentials. No retraining was performed, no dataset was regenerated, and no existing model/evaluation artifact was overwritten.

## Verdict

The simulator and its deterministic integration are functional and well-tested. Actual backend/frontend HTTP integration passed with a deterministic mock provider, while browser-level E2E and live model inference remain blocked by environment limitations. Semantic disclosure fidelity is demonstrated for the current closed grammar and adversarial fixtures, but broad semantic robustness is not established. It can honestly be presented as an educational simulator with guarded AI narration and a documented negative fine-tuning result; it should not be presented as clinically validated, production-ready, or as evidence that fine-tuning improved the base model.
