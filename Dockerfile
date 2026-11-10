
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install available Debian security updates
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --upgrade -r requirements.txt \
    && python -m pip install --no-cache-dir --upgrade "setuptools>=78.1.1"

# Copy application
COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
