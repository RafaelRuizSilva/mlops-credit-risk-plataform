"""Testes do entrypoint HTTP com fakes — sem MLflow, sem Postgres, sem rede.

O create_app recebe uma factory de AppServices: injetamos fakes e
exercitamos contrato HTTP, validação e mapeamento de erros.

Nota: TestClient PRECISA ser usado como context manager ('with') para o
lifespan (startup) rodar — fora dele o app.state nunca é populado.
"""
from contextlib import contextmanager
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from credit_risk_api.domain.exceptions import ModelUnavailableError, PredictionLogError
from credit_risk_api.domain.explanation import ExplanationService
from credit_risk_api.domain.models import CreditApplication, FeatureContribution, RiskScore
from credit_risk_api.domain.scoring import ScoringService
from credit_risk_api.entrypoints.app import AppServices, create_app

KNOWN_ID = UUID("00000000-0000-0000-0000-000000000001")

APPLICATION = CreditApplication(
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


class FakeModelProvider:
    def __init__(self, probability: float = 0.42):
        self._probability = probability

    def predict_probability(self, application: CreditApplication) -> float:
        return self._probability

    def model_version(self) -> str:
        return "fake-1"

    def explain(self, application: CreditApplication) -> tuple[FeatureContribution, ...]:
        return (
            FeatureContribution("n_late_90", 0.0, 1.2),
            FeatureContribution("age", 45.0, -0.3),
        )


class SpyPredictionLogger:
    def __init__(self):
        self.logged: list[tuple[CreditApplication, RiskScore]] = []

    def log(self, application: CreditApplication, score: RiskScore) -> None:
        self.logged.append((application, score))


class FailingPredictionLogger:
    def log(self, application: CreditApplication, score: RiskScore) -> None:
        raise PredictionLogError("banco indisponível")


class FakePredictionStore:
    def get(self, prediction_id: UUID) -> CreditApplication | None:
        return APPLICATION if prediction_id == KNOWN_ID else None


def _factory_que_falha() -> AppServices:
    raise ModelUnavailableError("registry fora do ar")


VALID_PAYLOAD = {
    "revolving_utilization": 0.77,
    "age": 45,
    "n_late_30_59": 2,
    "debt_ratio": 0.80,
    "monthly_income": 9120,
    "n_open_credit_lines": 13,
    "n_late_90": 0,
    "n_real_estate_loans": 6,
    "n_late_60_89": 0,
    "n_dependents": 2,
}


@contextmanager
def client_for(probability: float = 0.42, logger=None, services_factory=None):
    if services_factory is None:
        provider = FakeModelProvider(probability)
        services = AppServices(
            scoring=ScoringService(
                provider, logger if logger is not None else SpyPredictionLogger()
            ),
            explanation=ExplanationService(provider, FakePredictionStore()),
        )
        services_factory = lambda: services  # noqa: E731
    with TestClient(create_app(services_factory=services_factory)) as client:
        yield client


class TestCreatePrediction:
    def test_201_com_payload_valido(self):
        with client_for(probability=0.42) as client:
            resp = client.post("/predictions", json=VALID_PAYLOAD)
        assert resp.status_code == 201
        body = resp.json()
        assert body["probability_default"] == 0.42
        assert body["risk_band"] == "alto"
        assert body["model_version"] == "fake-1"
        assert "prediction_id" in body

    def test_renda_ausente_e_aceita(self):
        payload = {**VALID_PAYLOAD, "monthly_income": None}
        with client_for() as client:
            assert client.post("/predictions", json=payload).status_code == 201

    @pytest.mark.parametrize(
        "campo, valor",
        [
            ("age", 17),                      # menor de idade
            ("age", -1),
            ("revolving_utilization", -0.5),  # proporção negativa
            ("n_late_90", -1),
        ],
    )
    def test_422_para_valores_invalidos(self, campo, valor):
        payload = {**VALID_PAYLOAD, campo: valor}
        with client_for() as client:
            assert client.post("/predictions", json=payload).status_code == 422

    def test_422_para_campo_obrigatorio_ausente(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
        with client_for() as client:
            assert client.post("/predictions", json=payload).status_code == 422

    def test_503_quando_logger_falha(self):
        # fail-closed: predição sem auditoria não sai pela porta
        with client_for(logger=FailingPredictionLogger()) as client:
            assert client.post("/predictions", json=VALID_PAYLOAD).status_code == 503


class TestExplanation:
    def test_200_para_predicao_conhecida(self):
        with client_for() as client:
            resp = client.get(f"/predictions/{KNOWN_ID}/explanation")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_version"] == "fake-1"
        assert body["contributions"][0]["feature"] == "n_late_90"
        assert body["contributions"][0]["direction"] == "aumenta"
        assert body["contributions"][1]["direction"] == "reduz"

    def test_404_para_predicao_desconhecida(self):
        with client_for() as client:
            assert client.get(f"/predictions/{uuid4()}/explanation").status_code == 404

    def test_422_para_id_que_nao_e_uuid(self):
        with client_for() as client:
            assert client.get("/predictions/nao-e-uuid/explanation").status_code == 422


class TestHealthReady:
    def test_health_sempre_200(self):
        with client_for() as client:
            assert client.get("/health").status_code == 200

    def test_ready_200_com_modelo_carregado(self):
        with client_for() as client:
            assert client.get("/ready").status_code == 200

    def test_ready_503_sem_modelo(self):
        with client_for(services_factory=_factory_que_falha) as client:
            assert client.get("/ready").status_code == 503
            assert client.get("/health").status_code == 200  # liveness independe do modelo

    def test_predictions_503_sem_modelo(self):
        with client_for(services_factory=_factory_que_falha) as client:
            assert client.post("/predictions", json=VALID_PAYLOAD).status_code == 503
