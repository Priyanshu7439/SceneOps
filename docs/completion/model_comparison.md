# Fair base versus fine-tuned comparison

The preserved valid runs are:

- Base: `artifacts/evaluation/base_blind`
- Fine-tuned GGUF: `artifacts/evaluation/gguf_blind`

Both contain 1,181 completed predictions and identical dataset SHA-256 `20678f...1a1b`, task distribution 562 scene/619 temporal, temperature 0, max tokens 256, and system/user-only prompts. Failed generations were zero in both. The evaluator reports exact match, token precision/recall/F1, and sequence similarity identically.

| Task | Base F1 | Fine-tuned F1 | Fine-tuned minus base |
|---|---:|---:|---:|
| Overall | 0.6235269 | 0.5043650 | -11.9162 pp |
| Scene | 0.8181509 | 0.5649590 | -25.3192 pp |
| Temporal | 0.4468247 | 0.4493507 | +0.2526 pp |

Paired wins/losses/ties are 215/966/0 overall, 0/562/0 scene, and 215/404/0 temporal. The fine-tuned model is not better by lexical F1. The result is not a clinical or semantic correctness score; the independent state-fidelity report must be read separately.

The target-leaking `artifacts/evaluation/gguf_final` directory is preserved as historical evidence and excluded from all conclusions.
