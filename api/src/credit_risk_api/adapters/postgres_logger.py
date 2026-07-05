"""Adapter de saída: implementa a port PredictionLogger no PostgreSQL.

A tabela predictions é o registro de auditoria e a matéria-prima do
monitoramento de drift (Evidently, Fase 6). Features em JSONB: flexível
a mudanças de schema do modelo sem migração de coluna.
"""
import json
from dataclasses import asdict

import psycopg
from psycopg_pool import ConnectionPool

from credit_risk_api.domain.exceptions import PredictionLogError
from credit_risk_api.domain.models import CreditApplication, RiskScore

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id            uuid PRIMARY KEY,
    created_at    timestamptz NOT NULL DEFAULT now(),
    features      jsonb NOT NULL,
    probability   double precision NOT NULL,
    risk_band     text NOT NULL,
    model_version text NOT NULL
)
"""

_INSERT = """
INSERT INTO predictions (id, features, probability, risk_band, model_version)
VALUES (%(id)s, %(features)s, %(probability)s, %(risk_band)s, %(model_version)s)
"""


class PostgresPredictionLogger:
    def __init__(self, dsn: str):
        try:
            # open=True valida a conexão já no startup (fail-fast, como o modelo)
            self._pool = ConnectionPool(dsn, min_size=1, max_size=4, open=True)
            with self._pool.connection() as conn:
                conn.execute(_CREATE_TABLE)  # provisório; Alembic quando houver migrações
        except psycopg.Error as exc:
            raise PredictionLogError(f"falha ao preparar o banco de predições: {exc}") from exc

    def log(self, application: CreditApplication, score: RiskScore) -> None:
        try:
            with self._pool.connection() as conn:
                conn.execute(
                    _INSERT,
                    {
                        "id": score.prediction_id,
                        "features": json.dumps(asdict(application)),
                        "probability": score.probability_default,
                        "risk_band": score.risk_band.value,
                        "model_version": score.model_version,
                    },
                )
        except psycopg.Error as exc:
            raise PredictionLogError(f"falha ao registrar predição: {exc}") from exc

    def close(self) -> None:
        self._pool.close()
