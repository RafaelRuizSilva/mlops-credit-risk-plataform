"""Entidades e value objects do domínio.

Domínio puro: dataclasses do stdlib, sem FastAPI, sem Pydantic, sem MLflow.
Validação de formato (tipos, campos obrigatórios) é papel do entrypoint;
aqui moram apenas invariantes de negócio.
"""
from dataclasses import dataclass
from enum import Enum
from uuid import UUID


class RiskBand(str, Enum):
    BAIXO = "baixo"
    MEDIO = "medio"
    ALTO = "alto"

    @classmethod
    def from_probability(cls, probability: float) -> "RiskBand":
        # Faixas provisórias de política de crédito (taxa base ~7% de default).
        # Revisar com a curva de calibração antes de qualquer uso real.
        if probability < 0.10:
            return cls.BAIXO
        if probability < 0.30:
            return cls.MEDIO
        return cls.ALTO


@dataclass(frozen=True)
class CreditApplication:
    """Solicitação de crédito: as 10 features que o modelo conhece."""
    revolving_utilization: float
    age: int
    n_late_30_59: int
    debt_ratio: float
    monthly_income: float | None   # não declarada -> None; o pipeline imputa
    n_open_credit_lines: int
    n_late_90: int
    n_real_estate_loans: int
    n_late_60_89: int
    n_dependents: int | None


@dataclass(frozen=True)
class FeatureContribution:
    """Contribuição de uma feature para o score, em log-odds.

    Baseline = solicitante médio do treino; positivo empurra para o default.
    Para modelo linear isso equivale aos SHAP values com baseline na média.
    """
    feature: str
    value: float | None
    contribution: float


@dataclass(frozen=True)
class Explanation:
    prediction_id: UUID
    model_version: str
    contributions: tuple[FeatureContribution, ...]


@dataclass(frozen=True)
class RiskScore:
    """Resultado de uma avaliação de risco."""
    prediction_id: UUID
    probability_default: float
    risk_band: RiskBand
    model_version: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability_default <= 1.0:
            raise ValueError(
                f"probability_default deve estar em [0, 1], recebeu {self.probability_default}"
            )
