# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer-cached until pyproject.toml/uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Copy application code
COPY agent.py db.py ./
COPY tasks/ ./tasks/
COPY data/ ./data/

RUN chown -R appuser:appuser /app

USER appuser

# Pre-download Silero VAD / turn-detector model weights
RUN uv run agent.py download-files

CMD ["uv", "run", "agent.py", "start"]
