"""Treino do modelo de credit risk.

Uso: uv run python -m credit_risk.train
"""
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.data import clean, load_raw
from credit_risk.features import add_features

logger = logging.getLogger(__name__)

TARGET = "SeriousDlqin2yrs"


@dataclass(frozen=True)
class TrainConfig:
    """Tudo que é ajustável no treino, num lugar só."""
    test_size: float = 0.2
    n_splits: int = 5
    C: float = 1.0  # regularização da LogisticRegression
    random_state: int = 42


def build_pipeline(config: TrainConfig) -> Pipeline:
    """Pré-processamento + modelo. O objeto inteiro será serializado no MLflow."""
    return Pipeline([
        # atinge MonthlyIncome e NumberOfDependents; no-op nas colunas sem NaN
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            C=config.C,
            class_weight='balanced',
            max_iter=1000,
            random_state=config.random_state,
        )),
    ])


def split_data(
    df: pd.DataFrame, config: TrainConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa X/y e treino/teste, estratificado pelo target (classe rara ~7%)."""
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(
        X, y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )


def evaluate(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: TrainConfig,
) -> dict[str, float]:
    """Avalia com CV no treino + avaliação final no teste. Não imprime: devolve."""
    cv = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc')

    pipe.fit(X_train, y_train)
    proba_test = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba_test)

    return {
        'auc_cv_mean': float(np.mean(cv_scores)),
        'auc_cv_std': float(np.std(cv_scores)),
        'auc_test': float(roc_auc_score(y_test, proba_test)),
        'ks_test': float(np.max(tpr - fpr)),
    }


@dataclass(frozen=True)
class TrainResult:
    pipe: Pipeline
    metrics: dict[str, float]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.Series


def train(config: TrainConfig = TrainConfig()) -> TrainResult:
    """Orquestra: load -> clean -> features -> split -> pipeline -> evaluate."""
    logger.info("Carregando e preparando dados")
    df = add_features(clean(load_raw()))

    X_train, X_test, y_train, y_test = split_data(df, config)
    logger.info("Treino: %s | Teste: %s", X_train.shape, X_test.shape)

    pipe = build_pipeline(config)
    metrics = evaluate(pipe, X_train, y_train, X_test, y_test, config)
    logger.info("Métricas: %s", metrics)
    return TrainResult(pipe, metrics, X_train, X_test, y_test)


def run_experiment(
    config: TrainConfig = TrainConfig(), register: bool = False
) -> dict:
    """Envolve o treino com tracking: params, métricas, responsible AI e modelo.

    register=True também cria uma versão no Model Registry (usado pelo
    pipeline de treino; a promoção a champion é decisão de quem chama).
    Devolve run_id, métricas e a versão registrada (ou None).
    """
    from tempfile import TemporaryDirectory

    from credit_risk.responsible_ai import fairness_metrics, shap_summary_plot

    mlflow.set_experiment("credit-risk")
    with mlflow.start_run() as run:
        result = train(config)
        mlflow.log_params(asdict(config))       # o dataclass vira params de uma vez
        mlflow.log_metrics(result.metrics)      # o dict do evaluate cai direto

        fairness = fairness_metrics(result.pipe, result.X_test, result.y_test)
        mlflow.log_metrics(fairness)
        with TemporaryDirectory() as tmp:
            plot = shap_summary_plot(result.pipe, result.X_train, Path(tmp) / "shap_summary.png")
            mlflow.log_artifact(str(plot), artifact_path="explainability")

        model_info = mlflow.sklearn.log_model(
            result.pipe,                        # o Pipeline INTEIRO, com imputer e scaler
            name="model",
            input_example=result.X_train.head(5),
            # mlflow 3.x serializa com skops, que audita tipos por segurança;
            # numpy.dtype é usado internamente pelo StandardScaler e é seguro
            skops_trusted_types=["numpy.dtype"],
            registered_model_name="credit-risk-model" if register else None,
        )
        return {
            "run_id": run.info.run_id,
            "metrics": {**result.metrics, **fairness},
            "registered_version": model_info.registered_model_version if register else None,
        }


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    # console do Windows usa cp1252, que não representa os emojis dos logs do MLflow
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()  # MLFLOW_TRACKING_URI, credenciais do MinIO etc. (ml/.env)
    run_experiment()
