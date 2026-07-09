"""Schemas Pydantic: o contrato HTTP e sua tradução de/para o domínio.

Validação de FORMATO mora aqui (tipos, faixas, obrigatoriedade) e gera 422
automático. Regra de NEGÓCIO continua no domínio.
"""
from uuid import UUID

from pydantic import BaseModel, Field

from credit_risk_api.domain.models import CreditApplication, Explanation, RiskScore


class PredictionRequest(BaseModel):
    revolving_utilization: float = Field(ge=0)
    age: int = Field(ge=18, le=120)
    n_late_30_59: int = Field(ge=0)
    debt_ratio: float = Field(ge=0)
    monthly_income: float | None = Field(default=None, ge=0)
    n_open_credit_lines: int = Field(ge=0)
    n_late_90: int = Field(ge=0)
    n_real_estate_loans: int = Field(ge=0)
    n_late_60_89: int = Field(ge=0)
    n_dependents: int | None = Field(default=None, ge=0)

    def to_domain(self) -> CreditApplication:
        return CreditApplication(**self.model_dump())


class PredictionResponse(BaseModel):
    prediction_id: UUID
    probability_default: float
    risk_band: str
    model_version: str

    @classmethod
    def from_domain(cls, score: RiskScore) -> "PredictionResponse":
        return cls(
            prediction_id=score.prediction_id,
            probability_default=score.probability_default,
            risk_band=score.risk_band.value,
            model_version=score.model_version,
        )


class ContributionItem(BaseModel):
    feature: str
    value: float | None
    contribution_log_odds: float
    direction: str  # "aumenta" | "reduz" o risco


class ExplanationResponse(BaseModel):
    prediction_id: UUID
    model_version: str
    baseline: str
    contributions: list[ContributionItem]

    @classmethod
    def from_domain(cls, explanation: Explanation) -> "ExplanationResponse":
        return cls(
            prediction_id=explanation.prediction_id,
            model_version=explanation.model_version,
            baseline="solicitante médio do conjunto de treino (log-odds)",
            contributions=[
                ContributionItem(
                    feature=c.feature,
                    value=c.value,
                    contribution_log_odds=round(c.contribution, 4),
                    direction="aumenta" if c.contribution > 0 else "reduz",
                )
                for c in explanation.contributions
            ],
        )
