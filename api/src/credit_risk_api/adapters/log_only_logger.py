"""Adapter provisório da port PredictionLogger: registra no log da aplicação.

Existe para o M3 funcionar de ponta a ponta; o M4 o substitui pelo
PostgresPredictionLogger sem tocar em domínio ou entrypoints.
"""
import logging

from credit_risk_api.domain.models import CreditApplication, RiskScore

logger = logging.getLogger(__name__)


class LogOnlyPredictionLogger:
    def log(self, application: CreditApplication, score: RiskScore) -> None:
        logger.info(
            "prediction id=%s proba=%.4f band=%s model_version=%s",
            score.prediction_id,
            score.probability_default,
            score.risk_band.value,
            score.model_version,
        )
