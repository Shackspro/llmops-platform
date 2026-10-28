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


def test_generate_returns_model_response(monkeypatch):
    class FakeOllamaResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "Hello from a mocked model."}

    def fake_post(url, *, json, timeout):
        assert url == "http://127.0.0.1:11434/api/generate"
        assert json["model"] == "llama3.2:1b"
        assert json["prompt"] == "Say hello"
        assert json["stream"] is False

        return FakeOllamaResponse()

    monkeypatch.setattr("app.main.httpx.post", fake_post)

    response = client.post(
        "/generate",
        json={"prompt": "Say hello"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "model": "llama3.2:1b",
        "response": "Hello from a mocked model.",
    }


def test_generate_rejects_empty_prompt():
    response = client.post(
        "/generate",
        json={"prompt": ""},
    )

    assert response.status_code == 422