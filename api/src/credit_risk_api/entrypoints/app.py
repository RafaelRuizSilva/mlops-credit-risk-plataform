"""Montagem da aplicação: composição das dependências + tradução de erros.

Este é o ÚNICO lugar que conhece domínio E adapters concretos ao mesmo
tempo (composition root). Rodar em dev:

    uv run uvicorn credit_risk_api.entrypoints.app:app --reload --port 8000
"""
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from credit_risk_api.domain.exceptions import ModelUnavailableError, PredictionLogError
from credit_risk_api.domain.scoring import ScoringService
from credit_risk_api.entrypoints.routes import router

logger = logging.getLogger(__name__)


def _build_default_service() -> ScoringService:
    """Produção: adapters reais. Imports locais para testes não pagarem por eles."""
    import os

    from credit_risk_api.adapters.mlflow_provider import MLflowModelProvider
    from credit_risk_api.adapters.postgres_logger import PostgresPredictionLogger

    load_dotenv()
    dsn = os.environ["PREDICTIONS_DATABASE_URL"]
    return ScoringService(MLflowModelProvider(), PostgresPredictionLogger(dsn))


def create_app(
    service_factory: Callable[[], ScoringService] = _build_default_service,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.scoring_service = service_factory()
            logger.info("scoring service pronto")
        except ModelUnavailableError:
            # sobe mesmo assim: /health responde e /ready sinaliza 503
            logger.exception("champion indisponível no startup")
            app.state.scoring_service = None
        yield

    app = FastAPI(title="Credit Risk API", version="0.1.0", lifespan=lifespan)
    app.include_router(router)

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

    return app


app = create_app()
