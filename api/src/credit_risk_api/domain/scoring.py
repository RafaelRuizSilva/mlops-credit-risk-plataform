"""Caso de uso: avaliar o risco de uma solicitação de crédito."""
from uuid import uuid4

from credit_risk_api.domain.models import CreditApplication, RiskBand, RiskScore
from credit_risk_api.domain.ports import ModelProvider, PredictionLogger


class ScoringService:
    """Orquestra o fluxo: predição -> faixa de risco -> registro -> resposta.

    Decisão de consistência: se o registro da predição falhar, a avaliação
    falha junto (fail-closed). Em crédito, predição não auditável é passivo
    regulatório; preferimos indisponibilidade a resposta sem rastro.
    """

    def __init__(self, model_provider: ModelProvider, prediction_logger: PredictionLogger):
        self._model_provider = model_provider
        self._prediction_logger = prediction_logger

    def score(self, application: CreditApplication) -> RiskScore:
        probability = self._model_provider.predict_probability(application)
        risk_score = RiskScore(
            prediction_id=uuid4(),
            probability_default=probability,
            risk_band=RiskBand.from_probability(probability),
            model_version=self._model_provider.model_version(),
        )
        self._prediction_logger.log(application, risk_score)
        return risk_score
