"""Testes do use case com fakes das ports — sem Docker, sem rede, milissegundos.

Os fakes provam o valor da hexagonal: o domínio inteiro é exercitado
sem MLflow, Postgres ou FastAPI existirem.
"""
import pytest

from credit_risk_api.domain.exceptions import PredictionLogError
from credit_risk_api.domain.models import CreditApplication, RiskBand, RiskScore
from credit_risk_api.domain.scoring import ScoringService


class FakeModelProvider:
    """Satisfaz a port ModelProvider devolvendo uma probabilidade fixa."""

    def __init__(self, probability: float):
        self._probability = probability

    def predict_probability(self, application: CreditApplication) -> float:
        return self._probability

    def model_version(self) -> str:
        return "fake-1"


class SpyPredictionLogger:
    """Satisfaz a port PredictionLogger gravando as chamadas em memória."""

    def __init__(self):
        self.logged: list[tuple[CreditApplication, RiskScore]] = []

    def log(self, application: CreditApplication, score: RiskScore) -> None:
        self.logged.append((application, score))


class FailingPredictionLogger:
    def log(self, application: CreditApplication, score: RiskScore) -> None:
        # contrato de erros da port: o adapter traduz a exceção da tecnologia
        # (psycopg.Error etc.) para a exceção do DOMÍNIO antes de subir
        raise PredictionLogError("banco indisponível")


@pytest.fixture
def application() -> CreditApplication:
    return CreditApplication(
        revolving_utilization=0.77,
        age=45,
        n_late_30_59=2,
        debt_ratio=0.80,
        monthly_income=9120.0,
        n_open_credit_lines=13,
        n_late_90=0,
        n_real_estate_loans=6,
        n_late_60_89=0,
        n_dependents=2,
    )


def make_service(probability: float = 0.42, logger=None) -> tuple[ScoringService, SpyPredictionLogger]:
    logger = logger if logger is not None else SpyPredictionLogger()
    return ScoringService(FakeModelProvider(probability), logger), logger


class TestScore:
    def test_devolve_probabilidade_do_modelo(self, application):
        service, _ = make_service(probability=0.42)
        result = service.score(application)
        assert result.probability_default == 0.42

    def test_carrega_versao_do_modelo(self, application):
        service, _ = make_service()
        assert service.score(application).model_version == "fake-1"

    def test_gera_prediction_id_unico_por_avaliacao(self, application):
        service, _ = make_service()
        first = service.score(application)
        second = service.score(application)
        assert first.prediction_id != second.prediction_id

    def test_registra_a_predicao_no_logger(self, application):
        service, logger = make_service(probability=0.42)
        result = service.score(application)
        assert logger.logged == [(application, result)]

    def test_falha_do_logger_derruba_a_avaliacao(self, application):
        # fail-closed: predição sem registro de auditoria não é devolvida
        service, _ = make_service(logger=FailingPredictionLogger())
        with pytest.raises(PredictionLogError):
            service.score(application)


class TestRiskBand:
    @pytest.mark.parametrize(
        ("probability", "band"),
        [
            (0.0, RiskBand.BAIXO),
            (0.09, RiskBand.BAIXO),
            (0.10, RiskBand.MEDIO),   # fronteira: >= 0.10 é médio
            (0.29, RiskBand.MEDIO),
            (0.30, RiskBand.ALTO),    # fronteira: >= 0.30 é alto
            (1.0, RiskBand.ALTO),
        ],
    )
    def test_faixas(self, probability, band):
        assert RiskBand.from_probability(probability) is band

    def test_score_usa_a_faixa_da_probabilidade(self, application):
        service, _ = make_service(probability=0.05)
        assert service.score(application).risk_band is RiskBand.BAIXO


class TestRiskScoreInvariantes:
    @pytest.mark.parametrize("invalid", [-0.1, 1.1])
    def test_probabilidade_fora_de_0_1_e_rejeitada(self, application, invalid):
        service, _ = make_service(probability=invalid)
        with pytest.raises(ValueError, match="probability_default"):
            service.score(application)
