# Model serving

## Verified local runtime

Final local verification used llama.cpp with the existing GGUF through an OpenAI-compatible endpoint:

    $env:SCENEOPS_VLLM_BASE_URL = "http://127.0.0.1:8081/v1"
    $env:SCENEOPS_VLLM_MODEL = "auto"
    $env:PYTHONPATH = "src"
    python -m uvicorn sceneops.api.main:app --host 127.0.0.1 --port 8001

The provider model list returned one advertised model. Direct inference, backend scene narration, backend temporal narration, and frontend-proxy narration were verified. A dead-provider test returned HTTP 503 without changing the simulator snapshot.

## vLLM

src/sceneops/serving/vllm_backend.py is an OpenAI-compatible client. A standard vLLM deployment should use an available merged Hugging Face directory; the local GGUF must not be assumed to work with vLLM. Live vLLM server inference was not verified.

## Ray Serve

src/sceneops/serving/ray_serve_app.py provides health and generation routes, validation, timeout/error handling, and an injectable generator. Mocked health, failure, and timeout tests pass. Live model loading and inference remain unverified.

GGUF is intended for llama.cpp-compatible serving; merged Hugging Face is the candidate format for vLLM or Transformers; AWQ compatibility must be checked in the target environment.

## Request path and failure isolation

The application calls the provider through `src/sceneops/serving/vllm_backend.py`, which discovers the configured model, sends an OpenAI-compatible chat request, validates the response schema, and maps connection, HTTP, timeout, and malformed-response failures to safe application errors. The simulator state is not updated by this path. The Ray Serve adapter exposes the same conceptual health/generation boundary but is independently deployable and has not been represented as a verified production cluster.

The local evidence supports llama.cpp/GGUF inference. A vLLM deployment should use a compatible merged Hugging Face model directory when one is available; the presence of a GGUF file alone is not treated as vLLM compatibility.
