# Deployment-readiness checklist

This is a readiness audit only. No public deployment was attempted.

## Backend

- [x] FastAPI entry point: `sceneops.api.main:app`.
- [x] Health endpoint: `GET /health`.
- [x] Typed Pydantic contracts for session, simulator, narration, observation, resource, transport, and lifecycle routes.
- [x] Environment-based vLLM configuration: `SCENEOPS_VLLM_BASE_URL`, `SCENEOPS_VLLM_MODEL`, `SCENEOPS_VLLM_API_KEY`.
- [x] Error mapping for invalid requests, unavailable model responses, and invalid simulation transitions.
- [x] CORS is configurable through `SCENEOPS_CORS_ORIGINS`; default is no cross-origin allowance and wildcard CORS is not enabled.
- [ ] A live model endpoint is not available in this checkout.

Actual local HTTP integration passed with the FastAPI backend, Vite frontend, and deterministic mock provider. Browser automation was not completed because the browser access host was unavailable. The final Python suite is 530 passed with 4 warnings.

Startup:

```powershell
$env:SCENEOPS_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
$env:SCENEOPS_VLLM_MODEL = "sceneops"
\.venv\Scripts\python.exe -m uvicorn sceneops.api.main:app --host 127.0.0.1 --port 8001
```

## Ray Serve

- [x] Entry point: `scripts/serve_ray.py`.
- [x] `/health` and `/generate` validation, initialization errors, timeout, empty output, and model failures.
- [x] Ray-free mocked smoke tests.
- [ ] Live Ray/model inference remains unverified because Ray, torch, Transformers, and a loadable HF model directory are unavailable locally.

```powershell
python -m pip install "ray[serve]" torch transformers
\.venv\Scripts\python.exe scripts/serve_ray.py --model-id <merged-hf-model-directory>
Invoke-WebRequest http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/generate -Method Post -ContentType 'application/json' -Body '{"prompt":"Narrate approved observations only."}'
```

Shutdown: call `ray.serve.shutdown()` and then `ray.shutdown()`, or stop the serving process.

## vLLM

- [x] OpenAI-compatible async client and mocked HTTP tests.
- [ ] vLLM server not tested; vLLM is not installed.
- [ ] GGUF compatibility not assumed. The local artifact is GGUF Q4_K_M; use a supported merged Hugging Face directory for the standard vLLM path.

```bash
python -m pip install vllm
vllm serve /absolute/path/to/merged-hf-model --served-model-name sceneops --host 0.0.0.0 --port 8000
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"sceneops","messages":[{"role":"system","content":"Narrate only approved observations."},{"role":"user","content":"Patient P001: respiratory rate 24."}],"temperature":0,"max_tokens":128}'
```

The merged HF directory is not present locally, so server and model inference remain blocked.

### Verified local provider path

The available provider is llama.cpp serving the existing GGUF over the
OpenAI-compatible API at `http://127.0.0.1:8081/v1`. `/models` returned HTTP
200 and one advertised model ID; direct chat completion, scene narration, and
temporal narration through the backend returned HTTP 200. Configure:

```powershell
$env:SCENEOPS_VLLM_BASE_URL = "http://127.0.0.1:8081/v1"
$env:SCENEOPS_VLLM_MODEL = "auto"
```

`auto` is only a convenience for a provider exposing exactly one model. This
verifies the compatible client and llama.cpp provider, not a vLLM server.

## W&B

- [x] Disabled-by-default tracker, offline mode, dataset metadata, model metadata, scalar metrics, and clean termination.
- [x] Mocked tests pass without credentials.
- [ ] Remote logging unverified.

```powershell
$env:SCENEOPS_WANDB_MODE = "offline" # or online after authentication
python -m pip install wandb
wandb login
```

Remote logging requires a user-provided `WANDB_API_KEY`; it is never stored in source or printed.

## Axolotl

- [x] Optional configuration structurally validated against local model ID, dataset path, chat-template format, and LoRA settings.
- [ ] Actual Axolotl execution intentionally not performed; it would be a new training run.

```bash
python -m pip install axolotl
axolotl train configs/axolotl_sceneops_lora.yml
```

This is a future authorized command only and must not overwrite existing outputs.

## Frontend

- [x] `npm run build` passes.
- [x] Frontend uses relative `/sessions` and `/health` paths matching backend routes.
- [x] Vite proxy target is configurable through `VITE_API_PROXY_TARGET` and was verified against the backend on port 8001 (`/health` 200, `/sessions` 201).
- [x] Loading/error states exist for dashboard and narration operations.
- [x] No production code hardcodes a model URL or API key.
- [x] Vite's localhost proxy is development-only; production still needs same-origin serving or a reverse proxy to the backend.
- [ ] Browser-level E2E remains unverified because the browser automation host was unavailable.

## Gate

The repository is ready for separate frontend/backend end-to-end testing, with live model serving and production routing explicitly blocked until the runtime environment is supplied.
