# Model-serving architecture

The application depends on a provider abstraction rather than embedding model-runtime logic in the simulator.

```mermaid
flowchart LR
  APP[SceneOps application] --> CLIENT[VLLMBackend]
  CLIENT --> API[OpenAI-compatible models and chat completions]
  API --> LLAMA[llama.cpp plus GGUF, verified local runtime]
  CLIENT -. supported environment .-> VLLM[vLLM plus merged Hugging Face, client path]
  RAY[Ray Serve adapter] --> HEALTH[Health route]
  RAY --> GENERATE[Generation route]
  RAY -. optional deployment .-> HF[Loadable HF model directory]
  CLIENT --> TIMEOUT[Timeout and HTTP/schema errors]
  TIMEOUT --> APIERR[Safe API error; no simulator mutation]
```

llama.cpp is the verified local provider. The vLLM client is tested, but live vLLM inference and GGUF-through-vLLM are unverified. Ray Serve health, failure, and timeout behavior is mocked-tested; live model loading is unverified.
