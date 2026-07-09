"""Montagem da aplicação: composição das dependências + tradução de erros.

Este é o ÚNICO lugar que conhece domínio E adapters concretos ao mesmo
tempo (composition root). Rodar em dev:

    uv run uvicorn credit_risk_api.entrypoints.app:app --reload --port 8000
"""
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from credit_risk_api.domain.exceptions import (
    ModelUnavailableError,
    PredictionLogError,
    PredictionNotFoundError,
)
from credit_risk_api.domain.explanation import ExplanationService
from credit_risk_api.domain.scoring import ScoringService
from credit_risk_api.entrypoints.routes import router

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppServices:
    scoring: ScoringService
    explanation: ExplanationService


def _build_default_services() -> AppServices:
    """Produção: adapters reais. Imports locais para testes não pagarem por eles."""
    import os

    from credit_risk_api.adapters.mlflow_provider import MLflowModelProvider
    from credit_risk_api.adapters.postgres_logger import PostgresPredictionLogger

    load_dotenv()
    provider = MLflowModelProvider()
    store = PostgresPredictionLogger(os.environ["PREDICTIONS_DATABASE_URL"])
    return AppServices(
        scoring=ScoringService(provider, store),
        explanation=ExplanationService(provider, store),
    )


def create_app(
    services_factory: Callable[[], AppServices] = _build_default_services,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.services = services_factory()
            logger.info("serviços prontos (modelo carregado)")
        except ModelUnavailableError:
            # sobe mesmo assim: /health responde e /ready sinaliza 503
            logger.exception("champion indisponível no startup")
            app.state.services = None
        yield

    app = FastAPI(title="Credit Risk API", version="0.2.0", lifespan=lifespan)
    app.include_router(router)

    # /metrics para o Prometheus (latência, throughput, status por rota)
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app, include_in_schema=False)

    @app.exception_handler(ModelUnavailableError)
    async def model_unavailable(request: Request, exc: ModelUnavailableError):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "modelo indisponível, tente novamente"},
        )

    @app.exception_handler(PredictionLogError)
    async def prediction_log_error(request: Request, exc: PredictionLogError):
        # fail-closed (ver ScoringService): sem auditoria, sem resposta
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "não foi possível registrar a predição, tente novamente"},
        )

    @app.exception_handler(PredictionNotFoundError)
    async def prediction_not_found(request: Request, exc: PredictionNotFoundError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "predição não encontrada"},
        )

    return app


app = create_app()
