# Contributing

Keep simulator business rules deterministic and preserve the model boundary: language generation may not modify patient state, scoring, resources, transport, survival, or hidden disclosures.

Before opening a change:

1. Inspect git status and existing diffs.
2. Add focused tests for behavior changes.
3. Run pytest, Ruff, frontend lint, and frontend build.
4. Do not include .env files, credentials, model weights, checkpoints, private datasets, node_modules, or virtual environments.
5. Do not retrain or regenerate the preserved dataset during an application change.

Use synthetic fixtures only. This project is educational and not a clinical system.
