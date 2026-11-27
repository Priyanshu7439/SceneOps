# Development

Preserve simulator ground truth and experimental datasets. Do not retrain, regenerate datasets, overwrite models, or alter historical evaluation outputs during ordinary fixes.

    $env:PYTHONPATH = "src"
    python -m pytest -q
    python -m ruff check src tests
    npm run lint --prefix frontend
    npm run build --prefix frontend
    git diff --check

Use mocked providers in ordinary tests. Reserve real provider checks for controlled local smoke verification. Review secrets, paths, and generated artifacts before staging.
