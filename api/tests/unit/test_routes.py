"""Testes do entrypoint HTTP com fakes — sem MLflow, sem Postgres, sem rede.

O create_app recebe uma factory de ScoringService: injetamos fakes e
exercitamos contrato HTTP, validação e mapeamento de erros.

Nota: TestClient PRECISA ser usado como context manager ('with') para o
lifespan (startup) rodar — fora dele o app.state nunca é populado.
"""
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from credit_risk_api.domain.exceptions import ModelUnavailableError, PredictionLogError
from credit_risk_api.domain.models import CreditApplication, RiskScore
from credit_risk_api.domain.scoring import ScoringService
from credit_risk_api.entrypoints.app import create_app


class FakeModelProvider:
    def __init__(self, probability: float = 0.42):
        self._probability = probability

    def predict_probability(self, application: CreditApplication) -> float:
        return self._probability

    def model_version(self) -> str:
        return "fake-1"


class SpyPredictionLogger:
    def __init__(self):
        self.logged: list[tuple[CreditApplication, RiskScore]] = []

    def log(self, application: CreditApplication, score: RiskScore) -> None:
        self.logged.append((application, score))


class FailingPredictionLogger:
    def log(self, application: CreditApplication, score: RiskScore) -> None:
        raise PredictionLogError("banco indisponível")


def _factory_que_falha() -> ScoringService:
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
def client_for(probability: float = 0.42, logger=None, service_factory=None):
    if service_factory is None:
        service = ScoringService(
            FakeModelProvider(probability),
            logger if logger is not None else SpyPredictionLogger(),
        )
        service_factory = lambda: service  # noqa: E731
    with TestClient(create_app(service_factory=service_factory)) as client:
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


class TestHealthReady:
    def test_health_sempre_200(self):
        with client_for() as client:
            assert client.get("/health").status_code == 200

    def test_ready_200_com_modelo_carregado(self):
        with client_for() as client:
            assert client.get("/ready").status_code == 200

    def test_ready_503_sem_modelo(self):
        with client_for(service_factory=_factory_que_falha) as client:
            assert client.get("/ready").status_code == 503
            assert client.get("/health").status_code == 200  # liveness independe do modelo

    def test_predictions_503_sem_modelo(self):
        with client_for(service_factory=_factory_que_falha) as client:
            assert client.post("/predictions", json=VALID_PAYLOAD).status_code == 503
