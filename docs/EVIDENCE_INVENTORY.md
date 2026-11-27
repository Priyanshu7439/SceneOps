# Evidence inventory

| Claim | Evidence | Status |
|---|---|---|
| 5,856 examples / 320 scenarios | data/processed/sceneops_narration_manifest.json | Current checkout verified |
| 3,554 / 1,121 / 1,181 split | Dataset manifest | Current checkout verified |
| Scenario leakage audit passed | Manifest split_status and scenario lists | Current checkout verified |
| Blind model metrics | artifacts/evaluation/*_blind/metrics.json | Preserved completed runs inspected |
| 1,181/1,181 state fidelity | artifacts/evaluation/semantic/state_fidelity_v2_report.json | Preserved completed report inspected |
| QLoRA settings and 669 steps | docs/completion/training_audit.md, notebook_audit.txt | Historical evidence |
| GGUF artifact | models/gguf and completion audit | Artifact present; provenance not cryptographically proven |
| Real scene/temporal narration | final HTTP audit and provider smoke tests | Live llama.cpp verification |
| Ray Serve | serving implementation and tests | Mock-tested; live inference unverified |
| vLLM | client and tests | Client tested; vLLM server unverified |
| W&B | tracker and tests | Disabled/offline/mock-tested; remote unverified |
| Axolotl | config and validator/tests | Configuration validated; training not executed |
| Latest tests | current full suite | 532 passed, 5 warnings |
