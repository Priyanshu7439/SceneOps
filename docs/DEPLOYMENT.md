# Deployment

SceneOps is locally runnable and prepared for a separate deployment phase. Public deployment was not performed.

Backend:

    $env:PYTHONPATH = "src"
    python -m uvicorn sceneops.api.main:app --host 127.0.0.1 --port 8001

Frontend:

    cd frontend
    npm ci
    $env:VITE_API_PROXY_TARGET = "http://127.0.0.1:8001"
    npm run dev -- --host 127.0.0.1 --port 5173

Deployment requires a compatible provider/model, secret management, same-origin hosting or an explicit CORS allow-list, health checks, graceful shutdown, logging, and persistent storage if sessions must survive restarts. No production URL, throughput, latency, or uptime claim is made.
