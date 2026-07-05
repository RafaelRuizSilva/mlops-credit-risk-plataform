"""Adapter de entrada HTTP: desempacota, delega ao use case, empacota.

Router saudável é chato: nenhum if de negócio aqui.
"""
from fastapi import APIRouter, Depends, Request, Response, status

from credit_risk_api.domain.exceptions import ModelUnavailableError
from credit_risk_api.domain.scoring import ScoringService
from credit_risk_api.entrypoints.schemas import PredictionRequest, PredictionResponse

router = APIRouter()


def get_scoring_service(request: Request) -> ScoringService:
    service = request.app.state.scoring_service
    if service is None:
        raise ModelUnavailableError("modelo champion não carregado")
    return service


@router.post(
    "/predictions",
    status_code=status.HTTP_201_CREATED,
    response_model=PredictionResponse,
)
def create_prediction(
    payload: PredictionRequest,
    service: ScoringService = Depends(get_scoring_service),
) -> PredictionResponse:
    score = service.score(payload.to_domain())
    return PredictionResponse.from_domain(score)


@router.get("/health")
def health() -> dict:
    """Liveness: o processo responde."""
    return {"status": "ok"}


@router.get("/ready")
def ready(request: Request, response: Response) -> dict:
    """Readiness: o modelo está carregado e o serviço pode receber tráfego."""
    if request.app.state.scoring_service is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not ready", "reason": "modelo champion não carregado"}
    return {"status": "ready"}
