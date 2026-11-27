# Artifact inventory

Inventory scope: repository files and preserved uncommitted artifacts inspected on 2026-11-27. No files were deleted.

| Class | Examples | Decision |
|---|---|---|
| Required | `src/sceneops/`, `tests/`, processed narration splits, manifest, `README.md`, configs, `docs/completion/` | Retain and test |
| Historical evidence | `artifacts/evaluation/base_blind`, `gguf_blind`, `gguf_final`, `artifacts/final_audit/`, `notebook_audit.txt` | Retain; `gguf_final` is explicitly invalid for blind claims |
| Experimental but useful | `data/processed/patient_dialogue_v*.jsonl`, style data, dialogue generators, `notebooks/`, state-fidelity patch bundle | Retain per preservation rule |
| Safe cleanup candidate | `.tmp_pytest/` created by this audit; cache directories such as `.pytest_cache/` and `.ruff_cache/` | Review/remove locally after the run; not project artifacts |
| Unclear — retain | recovery patches, diff snapshots, duplicate top-level evaluator copies, frontend, zip bundles, debug/audit text files | User review required before cleanup |

The 986 MB GGUF, datasets, checkpoints if later restored, experimental dialogue material, and original evaluation outputs must not be deleted as “unused.”

New optional completion files include `scripts/mock_openai_server.py`, `scripts/serve_ray.py`, `src/sceneops/serving/ray_serve_app.py`, `src/sceneops/tracking/wandb_logging.py`, `src/sceneops/training/axolotl_config.py`, `.env.example` updates, and their tests. They are isolated and do not replace historical serving or training artifacts.

`frontend/.env.example` documents the configurable Vite backend proxy target; the prior hardcoded target mismatch was corrected rather than hidden.
