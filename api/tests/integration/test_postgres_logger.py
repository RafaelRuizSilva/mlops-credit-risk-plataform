"""Integração: o adapter PredictionLogger contra o Postgres real do compose."""
import os
from uuid import uuid4

import pytest

from credit_risk_api.domain.exceptions import PredictionLogError
from credit_risk_api.domain.models import CreditApplication, RiskBand, RiskScore
from credit_risk_api.adapters.postgres_logger import PostgresPredictionLogger

DSN = os.getenv(
    "PREDICTIONS_DATABASE_URL",
    "postgresql://mlflow:mlflow_dev_pw@localhost:5432/credit_risk",
)


@pytest.fixture(scope="module")
def logger_adapter():
    adapter = PostgresPredictionLogger(DSN)
    yield adapter
    adapter.close()


def _score(probability: float = 0.42) -> RiskScore:
    return RiskScore(
        prediction_id=uuid4(),
        probability_default=probability,
        risk_band=RiskBand.from_probability(probability),
        model_version="1",
    )


def _application() -> CreditApplication:
    return CreditApplication(
        revolving_utilization=0.2,
        age=45,
        n_late_30_59=0,
        debt_ratio=0.3,
        monthly_income=5000.0,
        n_open_credit_lines=5,
        n_late_90=0,
        n_real_estate_loans=1,
        n_late_60_89=0,
        n_dependents=1,
    )


def test_log_persiste_e_e_consultavel(logger_adapter):
    score = _score(probability=0.42)
    logger_adapter.log(_application(), score)

    with logger_adapter._pool.connection() as conn:
        row = conn.execute(
            "SELECT features, probability, risk_band, model_version, created_at"
            " FROM predictions WHERE id = %s",
            [score.prediction_id],
        ).fetchone()
        # limpeza: teste não deixa rastro
        conn.execute("DELETE FROM predictions WHERE id = %s", [score.prediction_id])

    assert row is not None
    features, probability, risk_band, model_version, created_at = row
    assert features["age"] == 45
    assert features["monthly_income"] == 5000.0
    assert probability == 0.42
    assert risk_band == "alto"
    assert model_version == "1"
    assert created_at is not None


def test_renda_none_vira_null_no_jsonb(logger_adapter):
    score = _score()
    application = CreditApplication(
        revolving_utilization=0.2, age=45, n_late_30_59=0, debt_ratio=0.3,
        monthly_income=None, n_open_credit_lines=5, n_late_90=0,
        n_real_estate_loans=1, n_late_60_89=0, n_dependents=None,
    )
    logger_adapter.log(application, score)

    with logger_adapter._pool.connection() as conn:
        (features,) = conn.execute(
            "SELECT features FROM predictions WHERE id = %s", [score.prediction_id]
        ).fetchone()
        conn.execute("DELETE FROM predictions WHERE id = %s", [score.prediction_id])

    assert features["monthly_income"] is None


def test_banco_inacessivel_vira_erro_de_dominio():
    dsn_invalido = "postgresql://mlflow:senha_errada@localhost:5432/credit_risk"
    with pytest.raises(PredictionLogError):
        PostgresPredictionLogger(dsn_invalido)
