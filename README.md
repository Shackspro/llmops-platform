# Local-First LLMOps Platform

A learning and portfolio project for building, testing,
containerizing, deploying, and monitoring an API for local
large language models.

## Current Features

- FastAPI application
- Root endpoint: GET /
- Health endpoint: GET /health
- Automated API tests using pytest

## Prerequisites

- Python
- Git

## Setup

Create and activate a virtual environment:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

Install dependencies:

    python -m pip install -r requirements.txt

## Run the API

    python -m uvicorn app.main:app --reload

API documentation:

    http://127.0.0.1:8000/docs

Health endpoint:

    http://127.0.0.1:8000/health

## Run Tests

    python -m pytest -v