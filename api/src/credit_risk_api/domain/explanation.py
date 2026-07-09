"""Caso de uso: explicar uma predição já registrada.

Regulação de crédito (ECOA/LGPD) exige justificar decisões — a explicação
opera sobre a predição AUDITADA (busca as features do registro), não sobre
um recálculo com dados novos.
"""
from uuid import UUID

from credit_risk_api.domain.exceptions import PredictionNotFoundError
from credit_risk_api.domain.models import Explanation
from credit_risk_api.domain.ports import ModelProvider, PredictionStore


class ExplanationService:
    def __init__(self, model_provider: ModelProvider, prediction_store: PredictionStore):
        self._model_provider = model_provider
        self._prediction_store = prediction_store

    def explain(self, prediction_id: UUID) -> Explanation:
        application = self._prediction_store.get(prediction_id)
        if application is None:
            raise PredictionNotFoundError(f"predição {prediction_id} não encontrada")
        return Explanation(
            prediction_id=prediction_id,
            model_version=self._model_provider.model_version(),
            contributions=self._model_provider.explain(application),
        )
