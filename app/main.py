
import hashlib
import json
import logging
import time
import uuid

import httpx
import redis
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import settings

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from fastapi.responses import Response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("llmops.api")

REQUEST_COUNT = Counter(
    "llmops_http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION = Histogram(
    "llmops_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

app = FastAPI(
    title="LLMOps Platform API",
    description="A local-first LLMOps learning project.",
    version="0.4.0",
)

redis_client = redis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    decode_responses=True,
    socket_connect_timeout=0.5,
    socket_timeout=0.5,
)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)


def enforce_rate_limit(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    rate_limit_key = f"rate-limit:{client_ip}"

    try:
        request_count = redis_client.incr(rate_limit_key)

        # Start the expiration timer on the first request.
        if request_count == 1:
            redis_client.expire(
                rate_limit_key,
                settings.rate_limit_window_seconds,
            )

    except redis.RedisError:
        # If Redis is unavailable, allow the request but log the issue.
        logger.warning("rate_limit_check_failed; allowing request")
        return

    if request_count > settings.rate_limit_requests:
        logger.warning(
            "rate_limit_exceeded client_ip=%s request_count=%s",
            client_ip,
            request_count,
        )
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            headers={
                "Retry-After": str(settings.rate_limit_window_seconds)
            },
        )



@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id

    started = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response

    finally:
        duration_seconds = time.perf_counter() - started

        # Use the registered route template to avoid
        # creating a separate metric for every arbitrary URL.
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()

        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration_seconds)

        logger.info(
            "request_completed request_id=%s method=%s path=%s "
            "status_code=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            status_code,
            duration_seconds * 1000,
        )



@app.get("/")
def read_root():
    return {"message": "LLMOps Platform API is running"}


@app.get("/health")
def health_check():
    # Retained for compatibility with earlier project milestones.
    return {"status": "healthy"}


@app.get("/health/live")
def liveness_check():
    # Indicates that the API process is responding.
    return {"status": "alive"}


@app.get("/health/ready")
def readiness_check():
    # Ollama is required to generate new responses.
    try:
        response = httpx.get(
            settings.ollama_tags_url,
            timeout=2.0,
        )
        response.raise_for_status()
        ollama_status = "up"
    except (httpx.HTTPError, ValueError):
        ollama_status = "down"

    # Redis is optional because inference can continue without caching.
    try:
        redis_client.ping()
        redis_status = "up"
    except redis.RedisError:
        redis_status = "degraded"

    if ollama_status != "up":
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "ollama": ollama_status,
                "redis": redis_status,
            },
        )

    return {
        "status": "ready",
        "ollama": ollama_status,
        "redis": redis_status,
    }

@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        content=generate_latest(),
        headers={"Content-Type": CONTENT_TYPE_LATEST},
    )


@app.post("/generate")
def generate_text(
        request: GenerateRequest,
        _: None = Depends(enforce_rate_limit),
):
    cache_key = "llm:" + hashlib.sha256(
        f"{settings.model_name}:{request.prompt}".encode("utf-8")
    ).hexdigest()

    try:
        cached = redis_client.get(cache_key)
        if cached is not None:
            result = json.loads(cached)
            result["cached"] = True
            return result
    except (redis.RedisError, json.JSONDecodeError):
        logger.warning("cache_read_failed; continuing without cache")

    try:
        response = httpx.post(
            settings.ollama_url,
            json={
                "model": settings.model_name,
                "prompt": request.prompt,
                "stream": False,
            },
            timeout=settings.request_timeout_seconds,
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
        "model": settings.model_name,
        "response": generated_text,
        "cached": False,
    }

    try:
        redis_client.setex(
            cache_key,
            settings.cache_ttl_seconds,
            json.dumps(result),
        )
    except redis.RedisError:
        logger.warning("cache_write_failed; returning generated response")

    return result
