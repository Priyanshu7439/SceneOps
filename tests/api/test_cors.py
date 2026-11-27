from fastapi.testclient import TestClient

from sceneops.api.main import create_app


def test_cors_allow_list_is_explicit(monkeypatch):
    monkeypatch.setenv("SCENEOPS_CORS_ORIGINS", "http://localhost:5173")
    app = create_app()

    with TestClient(app) as client:
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_does_not_default_to_wildcard(monkeypatch):
    monkeypatch.delenv("SCENEOPS_CORS_ORIGINS", raising=False)
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/health", headers={"Origin": "http://evil.test"})

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
