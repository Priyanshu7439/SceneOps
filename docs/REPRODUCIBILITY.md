# Reproducibility

## Local checks

    $env:PYTHONPATH = "src"
    python -m pytest -q
    python -m ruff check src tests
    npm run lint --prefix frontend
    npm run build --prefix frontend

The audited host required a writable temporary directory override for pytest because of permissions; this is not a project requirement.

## Application

Install requirements, copy .env.example to a local .env when needed, start a compatible provider, start FastAPI on port 8001, start Vite with VITE_API_PROXY_TARGET=http://127.0.0.1:8001, create a session, record observations, and exercise scene/temporal narration.

Historical training cannot be fully reproduced from this checkout because the original checkpoint and external GPU environment are unavailable.
