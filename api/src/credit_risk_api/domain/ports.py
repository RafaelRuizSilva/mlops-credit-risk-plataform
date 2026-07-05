"""Ports: as interfaces que o domínio exige do mundo externo.

O domínio declara O QUE precisa; os adapters (MLflow, Postgres...) decidem
COMO entregar. Protocol = duck typing verificável: qualquer classe com esses
métodos satisfaz a port, sem herdar de nada.
"""
from typing import Protocol

from credit_risk_api.domain.models import CreditApplication, RiskScore


class ModelProvider(Protocol):
    def predict_probability(self, application: CreditApplication) -> float:
        """Probabilidade de default em [0, 1] para a solicitação."""
        ...

    def model_version(self) -> str:
        """Identifica o modelo que responde (rastreabilidade/auditoria)."""
        ...


class PredictionLogger(Protocol):
    def log(self, application: CreditApplication, score: RiskScore) -> None:
        """Persiste a predição para auditoria e monitoramento de drift."""
        ...
