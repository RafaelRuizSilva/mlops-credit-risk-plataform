"""Adapter de saída: implementa a port ModelProvider via MLflow Model Registry.

Toda a tecnologia (mlflow, pandas, nomes de coluna do treino) fica confinada
aqui. Para o domínio, isto é apenas "algo que satisfaz ModelProvider".
"""
import math

import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from credit_risk_api.domain.exceptions import ModelUnavailableError
from credit_risk_api.domain.models import CreditApplication


def _to_feature_frame(app: CreditApplication) -> pd.DataFrame:
    """Traduz a entidade do domínio para o contrato do modelo (signature).

    Nomes e ordem das colunas são os do treino (ml/src/credit_risk).
    """
    return pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": app.revolving_utilization,
        "age": app.age,
        "NumberOfTime30-59DaysPastDueNotWorse": app.n_late_30_59,
        "DebtRatio": app.debt_ratio,
        "MonthlyIncome": math.nan if app.monthly_income is None else app.monthly_income,
        "NumberOfOpenCreditLinesAndLoans": app.n_open_credit_lines,
        "NumberOfTimes90DaysLate": app.n_late_90,
        "NumberRealEstateLoansOrLines": app.n_real_estate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": app.n_late_60_89,
        "NumberOfDependents": math.nan if app.n_dependents is None else app.n_dependents,
        # ATENÇÃO: espelha ml/features.add_features à mão -> risco de
        # training/serving skew; unificar quando a feature store entrar (Fase 5)
        "declarou_renda": 0 if app.monthly_income is None else 1,
    }])


class MLflowModelProvider:
    """Carrega o champion UMA vez (no construtor) e prediz em memória."""

    def __init__(self, model_name: str = "credit-risk-model", alias: str = "champion"):
        model_uri = f"models:/{model_name}@{alias}"
        try:
            self._pipeline = mlflow.sklearn.load_model(model_uri)
            self._version = MlflowClient().get_model_version_by_alias(model_name, alias).version
        except Exception as exc:  # traduz QUALQUER falha de infra para o domínio
            raise ModelUnavailableError(f"falha ao carregar {model_uri}: {exc}") from exc

    def predict_probability(self, application: CreditApplication) -> float:
        features = _to_feature_frame(application)
        return float(self._pipeline.predict_proba(features)[0, 1])

    def model_version(self) -> str:
        return self._version
