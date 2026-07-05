"""Treino do modelo de credit risk.

Uso: uv run python -m credit_risk.train
"""
import logging
from dataclasses import dataclass

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


def train(config: TrainConfig = TrainConfig()) -> dict[str, float]:
    """Orquestra: load -> clean -> features -> split -> pipeline -> evaluate."""
    logger.info("Carregando e preparando dados")
    df = add_features(clean(load_raw()))

    X_train, X_test, y_train, y_test = split_data(df, config)
    logger.info("Treino: %s | Teste: %s", X_train.shape, X_test.shape)

    pipe = build_pipeline(config)
    metrics = evaluate(pipe, X_train, y_train, X_test, y_test, config)
    logger.info("Métricas: %s", metrics)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train()
