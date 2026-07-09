"""Monitoramento de drift: predições em produção vs. dados de treino.

Roda em batch (agendável):
    uv run python flows/drift_flow.py

Lê a tabela de auditoria (predictions.features JSONB) como dado "current",
o dataset de treino como "reference", e gera o relatório do Evidently em
reports/drift_report.html. Em produção real, as métricas iriam também para
o Prometheus; aqui o HTML já cumpre o papel de estudo.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset
from prefect import flow, task, get_run_logger

from credit_risk.data import clean, load_raw
from credit_risk.features import add_features

# treino -> nomes do contrato da API (os mesmos gravados no JSONB)
TRAIN_TO_API = {
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "n_late_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "n_open_credit_lines",
    "NumberOfTimes90DaysLate": "n_late_90",
    "NumberRealEstateLoansOrLines": "n_real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "n_late_60_89",
    "NumberOfDependents": "n_dependents",
}

REPORT_PATH = Path(__file__).parents[1] / "reports" / "drift_report.html"


@task
def load_reference(sample: int = 5000) -> pd.DataFrame:
    df = add_features(clean(load_raw())).drop(columns="SeriousDlqin2yrs")
    df = df.rename(columns=TRAIN_TO_API)
    return df.sample(n=min(sample, len(df)), random_state=42)


@task
def load_current() -> pd.DataFrame:
    dsn = os.environ["PREDICTIONS_DATABASE_URL"]
    with psycopg.connect(dsn) as conn:
        rows = conn.execute("SELECT features FROM predictions").fetchall()
    df = pd.DataFrame([r[0] for r in rows])
    if not df.empty:
        df["declarou_renda"] = df["monthly_income"].notna().astype(int)
    return df


@task
def build_report(reference: pd.DataFrame, current: pd.DataFrame) -> Path:
    report = Report([DataDriftPreset()])
    snapshot = report.run(current_data=current[reference.columns], reference_data=reference)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(REPORT_PATH))
    return REPORT_PATH


@flow(name="credit-risk-drift")
def drift_pipeline() -> str:
    logger = get_run_logger()
    current = load_current()
    if len(current) < 2:
        logger.warning("Poucas predições (%s) para análise de drift", len(current))
        return "SEM DADOS SUFICIENTES"
    reference = load_reference()
    path = build_report(reference, current)
    logger.info("Relatório em %s (%s predições analisadas)", path, len(current))
    return f"OK: {path}"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    print(drift_pipeline())
