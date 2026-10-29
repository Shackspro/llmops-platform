import hashlib
import json
import os

import httpx
import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="LLMOps Platform API",
    description="A local-first LLMOps learning project.",
    version="0.3.0",
)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/api/generate",
)
MODEL_NAME = "llama3.2:1b"
CACHE_TTL_SECONDS = 3600

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    decode_responses=True,
    socket_connect_timeout=0.5,
    socket_timeout=0.5,
)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)


@app.get("/")
def read_root():
    return {"message": "LLMOps Platform API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/generate")
def generate_text(request: GenerateRequest):
    cache_key = "llm:" + hashlib.sha256(
        f"{MODEL_NAME}:{request.prompt}".encode("utf-8")
    ).hexdigest()

    # 1. Try Redis. If it is unavailable, continue without caching.
    try:
        cached = redis_client.get(cache_key)
        if cached is not None:
            result = json.loads(cached)
            result["cached"] = True
            return result
    except (redis.RedisError, json.JSONDecodeError):
        pass

    # 2. Cache miss: ask Ollama to generate a response.
    try:
        response = httpx.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": request.prompt,
                "stream": False,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        generated_text = response.json().get("response")

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not reachable.",
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
    except (ValueError, httpx.DecodingError):
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an invalid response.",
        ) from None

    if not isinstance(generated_text, str):
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an invalid response.",
        )

    result = {
        "model": MODEL_NAME,
        "response": generated_text,
        "cached": False,
    }

    # 3. Store the result for one hour. Redis failure must not
    #    prevent an otherwise successful model response.
    try:
        redis_client.setex(
            cache_key,
            CACHE_TTL_SECONDS,
            json.dumps(result),
        )
    except redis.RedisError:
        pass

    return result
