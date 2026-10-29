
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_returns_welcome_message():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "LLMOps Platform API is running"
    }


def test_health_returns_healthy_status():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_liveness():
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_request_id_is_returned():
    response = client.get(
        "/health/live",
        headers={"X-Request-ID": "test-request-123"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request-123"


def test_readiness(monkeypatch):
    class FakeOllamaResponse:
        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        "app.main.httpx.get",
        lambda *args, **kwargs: FakeOllamaResponse(),
    )
    monkeypatch.setattr(
        "app.main.redis_client.ping",
        lambda: True,
    )

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "ollama": "up",
        "redis": "up",
    }


def test_generate_returns_model_response(monkeypatch):
    class FakeOllamaResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "Hello from a mocked model."}

    monkeypatch.setattr(
        "app.main.httpx.post",
        lambda *args, **kwargs: FakeOllamaResponse(),
    )
    monkeypatch.setattr(
        "app.main.redis_client.get",
        lambda key: None,
    )
    monkeypatch.setattr(
        "app.main.redis_client.setex",
        lambda *args: True,
    )
    monkeypatch.setattr("app.main.redis_client.incr", lambda key: 1)
    monkeypatch.setattr(
        "app.main.redis_client.expire",
        lambda *args: True,
    )

    response = client.post(
        "/generate",
        json={"prompt": "Say hello"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "model": "llama3.2:1b",
        "response": "Hello from a mocked model.",
        "cached": False,
    }


def test_generate_rejects_empty_prompt():
    response = client.post(
        "/generate",
        json={"prompt": ""},
    )

    assert response.status_code == 422


def test_generate_returns_429_when_rate_limit_is_exceeded(monkeypatch):
    monkeypatch.setattr(
        "app.main.redis_client.incr",
        lambda key: 11,
    )

    response = client.post(
        "/generate",
        json={"prompt": "Say hello"},
    )

    assert response.status_code == 429
    assert response.json()["detail"] == (
        "Rate limit exceeded. Please try again later."
    )
    assert response.headers["Retry-After"] == "60"

