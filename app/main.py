import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="LLMOps Platform API",
    description="A local-first LLMOps learning project.",
    version="0.2.0",
)

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "llama3.2:1b"


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)


@app.get("/")
def read_root():
    return {
        "message": "LLMOps Platform API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/generate")
def generate_text(request: GenerateRequest):
    payload = {
        "model": MODEL_NAME,
        "prompt": request.prompt,
        "stream": False,
    }

    try:
        response = httpx.post(
            OLLAMA_URL,
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not reachable. Make sure it is running.",
        ) from None

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="The model took too long to respond.",
        ) from None

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned HTTP {exc.response.status_code}.",
        ) from None

    generated_text = response.json().get("response")

    if not isinstance(generated_text, str):
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an invalid response.",
        )

    return {
        "model": MODEL_NAME,
        "response": generated_text,
    }