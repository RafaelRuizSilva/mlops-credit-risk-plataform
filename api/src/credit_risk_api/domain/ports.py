"""Ports: as interfaces que o domínio exige do mundo externo.

O domínio declara O QUE precisa; os adapters (MLflow, Postgres...) decidem
COMO entregar. Protocol = duck typing verificável: qualquer classe com esses
métodos satisfaz a port, sem herdar de nada.
"""
from typing import Protocol
from uuid import UUID

from credit_risk_api.domain.models import CreditApplication, FeatureContribution, RiskScore


class ModelProvider(Protocol):
    def predict_probability(self, application: CreditApplication) -> float:
        """Probabilidade de default em [0, 1] para a solicitação.

        Levanta ModelUnavailableError se o modelo não puder responder.
        """
        ...

    def model_version(self) -> str:
        """Identifica o modelo que responde (rastreabilidade/auditoria)."""
        ...

    def explain(self, application: CreditApplication) -> tuple[FeatureContribution, ...]:
        """Contribuição de cada feature para o score, da maior para a menor (|contrib|)."""
        ...


class PredictionStore(Protocol):
    def get(self, prediction_id: UUID) -> CreditApplication | None:
        """Recupera as features de uma predição registrada (None se não existir)."""
        ...


class PredictionLogger(Protocol):
    def log(self, application: CreditApplication, score: RiskScore) -> None:
        """Persiste a predição para auditoria e monitoramento de drift.

        Levanta PredictionLogError se não conseguir persistir — o contrato
        de erros faz parte da port: adapters traduzem exceções da sua
        tecnologia para esta antes de deixá-las subir.
        """
        ...
