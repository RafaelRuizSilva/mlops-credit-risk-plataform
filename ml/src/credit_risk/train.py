"""Treino do modelo de credit risk.

Uso: uv run python -m credit_risk.train
"""
import logging
from dataclasses import asdict, dataclass

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


def train(config: TrainConfig = TrainConfig()) -> tuple[Pipeline, dict[str, float], pd.DataFrame]:
    """Orquestra: load -> clean -> features -> split -> pipeline -> evaluate.

    Devolve também um exemplo de entrada (linhas do X_train) para o MLflow
    inferir a assinatura do modelo — o contrato que a API validará.
    """
    logger.info("Carregando e preparando dados")
    df = add_features(clean(load_raw()))

    X_train, X_test, y_train, y_test = split_data(df, config)
    logger.info("Treino: %s | Teste: %s", X_train.shape, X_test.shape)

    pipe = build_pipeline(config)
    metrics = evaluate(pipe, X_train, y_train, X_test, y_test, config)
    logger.info("Métricas: %s", metrics)
    return pipe, metrics, X_train.head(5)


def run_experiment(config: TrainConfig = TrainConfig()) -> None:
    """Envolve o treino com tracking: params, métricas e modelo numa run do MLflow."""
    mlflow.set_experiment("credit-risk")
    with mlflow.start_run():
        pipe, metrics, input_example = train(config)
        mlflow.log_params(asdict(config))   # o dataclass vira params de uma vez
        mlflow.log_metrics(metrics)         # o dict do evaluate cai direto
        mlflow.sklearn.log_model(
            pipe,                           # o Pipeline INTEIRO, com imputer e scaler
            name="model",
            input_example=input_example,
            # mlflow 3.x serializa com skops, que audita tipos por segurança;
            # numpy.dtype é usado internamente pelo StandardScaler e é seguro
            skops_trusted_types=["numpy.dtype"],
        )


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    # console do Windows usa cp1252, que não representa os emojis dos logs do MLflow
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()  # MLFLOW_TRACKING_URI, credenciais do MinIO etc. (ml/.env)
    run_experiment()
