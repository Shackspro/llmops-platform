from fastapi import FastAPI

app = FastAPI(
    title="LLMOps Platform API",
    description="A local-first LLMOps learning project.",
    version="0.1.0",
)


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