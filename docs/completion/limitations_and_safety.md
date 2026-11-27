# Limitations and safety

- SceneOps is an educational synthetic simulation, not a clinically validated diagnostic, triage, treatment, or survival system.
- Python simulator logic owns patient state, deterioration, triage, resource allocation, transport, scoring, and outcomes. The LLM is limited to language generation from authorized disclosures.
- State-fidelity evaluation is template-bound and does not establish general semantic understanding, clinical correctness, or robustness to every prompt attack.
- Blind token F1 shows a substantial fine-tuned regression, especially for scene narration. Fine-tuning should not be called successful from training completion alone.
- The local GGUF's size and timestamp match historical conversion evidence, but no original Kaggle SHA-256 was recorded; cryptographic provenance is unverified.
- Ray Serve, vLLM live serving, W&B remote logging, and Axolotl execution are not live-verified in this checkout.
- Production frontend routing still requires same-origin hosting or a reviewed CORS/reverse-proxy configuration; the Vite localhost proxy is development-only. The backend now supports an explicit `SCENEOPS_CORS_ORIGINS` allow-list.
- Live Ray Serve and vLLM inference require a separately provisioned compatible runtime and model directory; their absence here is an environment blocker, not a passing deployment result.
- The project contains large historical artifacts, experimental dialogue datasets/code, and prior evaluation runs. They are intentionally retained pending an explicit cleanup decision.
