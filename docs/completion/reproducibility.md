# Reproducibility

## Local simulator and tests

```powershell
New-Item -ItemType Directory -Force .tmp_pytest | Out-Null
$env:TEMP=(Resolve-Path .tmp_pytest)
$env:TMP=$env:TEMP
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
```

The writable temp override is needed on the audited host because its default Windows pytest temp directory is access-denied. It is an environment workaround, not a project requirement.

## State fidelity

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_sceneops_state_fidelity_v2.py `
  --mapping artifacts/final_audit/sceneops_test_ground_truth_mapping.jsonl `
  --output-dir artifacts/evaluation/semantic
```

## Existing blind artifacts

Do not overwrite `artifacts/evaluation/base_blind`, `gguf_blind`, or `gguf_final`. A new endpoint comparison should write to a new directory and retain the same test manifest, prompt-only policy, temperature 0, max tokens 256, and evaluator version.

## Optional serving

Ray Serve requires an explicit optional environment with `ray[serve]`, torch, and Transformers. The checked-in adapter exposes `/health` and `/generate`; use `build_deployment(model_id=...)` or inject a generator for tests. Shutdown is `ray.serve.shutdown()` followed by `ray.shutdown()`.

vLLM is currently a client integration in `src/sceneops/serving/vllm_backend.py`. The GGUF artifact was not live-tested with vLLM in this environment; do not assume direct GGUF compatibility.

For the supported vLLM path, install `vllm` in a compatible Linux/CUDA environment, serve an available merged Hugging Face model directory with `vllm serve`, verify `/v1/models`, then send an OpenAI-compatible `/v1/chat/completions` request. The merged Hugging Face directory is not present locally.

## Optional tracking and Axolotl

`src/sceneops/tracking/wandb_logging.py` is disabled by default and supports offline mode, dataset split metadata, model/adapter metadata, scalar training/evaluation metrics, and clean termination. Set `SCENEOPS_WANDB_MODE=offline`, install `wandb`, and use the project wrapper; use `wandb login` plus `SCENEOPS_WANDB_MODE=online` only when remote logging is intentionally enabled. The Axolotl YAML is structurally validated by `src/sceneops/training/axolotl_config.py`; `axolotl train configs/axolotl_sceneops_lora.yml` remains an unexecuted future training command.
