# --- estágio 1: resolver e instalar dependências com uv -----------------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# 1º só os manifests: esta camada só rebuilda se as DEPS mudarem (cache!)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# depois o código: mudar código não invalida a camada de dependências
COPY src ./src
RUN uv sync --frozen --no-dev

# --- estágio 2: imagem final enxuta, sem uv e sem toolchain --------------
FROM python:3.11-slim-bookworm
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src ./src
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "credit_risk_api.entrypoints.app:app", "--host", "0.0.0.0", "--port", "8000"]
