# Final change audit

Audit date: 2026-11-27. Branch: `main`, HEAD `f8732ab`. The working tree was already dirty when this pass began; all pre-existing edits and untracked artifacts were preserved.

## Current architecture

- `src/sceneops/simulator/` remains the deterministic source of patient, observation, deterioration, triage, resource, transport, and outcome state.
- `src/sceneops/application/service.py` builds narration requests from simulator-approved observations and does not apply model output to simulator state.
- `src/sceneops/agents/` contains typed scene, temporal, dialogue, orchestration, and rule-based safety boundaries.
- `src/sceneops/api/main.py` exposes the FastAPI application and environment-configured model client.
- `frontend/` is a Vite/TypeScript application using relative `/sessions` and `/health` routes.
- `src/sceneops/serving/` contains the vLLM-compatible client and optional Ray Serve adapter.

## Changes made this final pass

- Added `scripts/mock_openai_server.py` for deterministic local-only HTTP integration testing; it is not a production model provider.
- Added explicit, opt-in CORS allow-list configuration using `SCENEOPS_CORS_ORIGINS`; wildcard CORS is not enabled.
- Added `tests/api/test_cors.py`.
- Added Ray Serve timeout/startup failure handling and tested its dependency-light generation boundary.
- Added W&B disabled/offline/mock lifecycle and metadata tests.
- Added Axolotl structural validation and test coverage without starting training.
- Added simulator-boundary tests for model failure and state immutability.
- Added `docs/completion/deployment_readiness.md` updates and this audit.
- Fixed a real frontend integration defect: Vite was proxying to port 8000 while the documented backend runs on 8001. The target is now configurable through `VITE_API_PROXY_TARGET`, defaulting to `http://127.0.0.1:8001`.

## Actual application execution

Started simultaneously:

- Mock OpenAI-compatible provider: `127.0.0.1:8010`.
- FastAPI backend: `127.0.0.1:8001`.
- Vite frontend: `http://localhost:5173`.

Real HTTP workflow passed against the actual backend and frontend servers: health, session creation, session start, observation recording, scene narration, temporal narration, assessment, ambulance allocation, transport start, and time advancement. The mock provider returned deterministic text.

The mock provider was stopped and a real backend narration request returned HTTP 503. Before/after session snapshots showed unchanged clock (`60` seconds) and event count (`27`), confirming model unavailability did not mutate simulator state.

The browser automation host reported unavailable, so no browser-level click/navigation test or screenshot claim is made. Frontend HTTP serving and production build were verified.

After the proxy fix, Vite `/health` returned HTTP 200 and proxied `/sessions` returned HTTP 201 against the running backend.

## Verification results

- Full Python suite: 530 passed, 0 failed, 0 skipped.
- Ruff: clean.
- Frontend production build: passed.
- Focused API/application/serving/tracking/training tests: 97 passed.
- Frontend root HTTP response: 200.
- Backend `/health`: 200.
- Configured CORS origin: `http://localhost:5173`.
- WSL llama.cpp attempt: blocked by `E_ACCESSDENIED`; no live llama.cpp inference claim.

## Preserved

No dataset, model artifact, checkpoint, original evaluation output, simulator ground-truth behavior, experimental patient-dialogue code, or user-created recovery artifact was overwritten or deleted. No retraining, commit, push, or public deployment was performed.

## vLLM narration incident repair

The 503 root cause was configuration: port 8000 was unreachable and the
placeholder model was not the model advertised by the live provider. The
llama.cpp OpenAI-compatible server at `http://127.0.0.1:8081/v1/models`
returned HTTP 200 and advertised the loaded GGUF under its full WSL path ID;
direct `/v1/chat/completions` inference returned the expected
`choices[0].message.content` schema.

`VLLMBackend` now resolves a sole advertised model for `auto`/placeholder
configuration, sends the resolved ID, and reports endpoint/schema failures
without exposing credentials. The temporal prompt no longer includes a
misleading example that caused the small model to copy unauthorized timestamps
and measurements. Real backend scene and temporal narration both returned HTTP
200 using the live provider. A dead-provider run returned HTTP 503 and left the
session snapshot unchanged.

Final relevant checks: 532 Python tests passed, frontend lint and production
build passed, and Ruff is clean for `src` and `tests`.

## Temporal narration regression repair

The provider was not offline. The temporal model response was being rejected
because the frontend asked for a change summary while the temporal prompt and
safety parser required one exact sentence per timestamp. The small GGUF model
also copied range-style/change wording that did not match the reviewer’s
measurement grammar. The temporal agent now sends an authoritative formatting
instruction and supplies canonical rendering targets derived only from the
authorized observations. No safety rule was disabled.

Real verification through the frontend Vite proxy and backend used observations
for P005 at seconds 0, 60, and 120. Scene narration and temporal narration both
returned HTTP 200; temporal output preserved all three timestamps and values.
The session clock, event count, and resource state were unchanged by narration.
