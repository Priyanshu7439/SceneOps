# Evaluation

The authoritative blind runs are artifacts/evaluation/base_blind and artifacts/evaluation/gguf_blind. Both evaluated 1,181 examples with the same dataset SHA, temperature 0, max tokens 256, and system/user-only prompts with assistant targets excluded.

| Metric | Base | Fine-tuned GGUF | Delta |
|---|---:|---:|---:|
| Overall token F1 | 0.6235269 | 0.5043650 | -11.9162 pp |
| Scene token F1 | 0.8181509 | 0.5649590 | -25.3192 pp |
| Temporal token F1 | 0.4468247 | 0.4493507 | +0.2526 pp |

Fine-tuned wins/losses/ties were 215/966/0 overall, 0/562/0 scene, and 215/404/0 temporal. Fine-tuning did not improve preserved lexical F1. Lexical F1 is not semantic, safety, or clinical correctness.

The independent state-fidelity report records 1,181/1,181 under a closed-grammar evaluator using regenerated simulator observations. It checks identity, timestamps, coverage, and values; it does not establish clinical safety or broad free-form understanding.

The older artifacts/evaluation/gguf_final output contains target leakage and is excluded from conclusions.
