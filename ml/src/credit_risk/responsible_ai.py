"""Explicabilidade global (SHAP) e fairness (Fairlearn) — etapas do treino.

Ambos consomem o pipeline treinado + dados de teste e produzem:
- um summary plot SHAP (artefato para o MLflow);
- métricas de disparidade por faixa etária (idade é o atributo sensível
  disponível neste dataset; ECOA/LGPD exigem monitorar disparidade).
"""
from pathlib import Path

import matplotlib
import pandas as pd
import shap
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")  # sem display no pipeline
import matplotlib.pyplot as plt  # noqa: E402


def shap_summary_plot(pipe: Pipeline, X: pd.DataFrame, out_path: Path, sample: int = 500) -> Path:
    """SHAP global: LinearExplainer para modelos lineares, TreeExplainer para árvores."""
    Xs = X.sample(n=min(sample, len(X)), random_state=42)
    model = pipe[-1]
    X_input = pipe[:-1].transform(Xs) if len(pipe.steps) > 1 else Xs
    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, X_input)
    else:
        explainer = shap.TreeExplainer(model)
    values = explainer(X_input)
    if values.values.ndim == 3:      # binário em árvores: pega a classe positiva
        values = values[:, :, 1]
    values.feature_names = list(Xs.columns)

    plt.figure()
    shap.plots.beeswarm(values, show=False, max_display=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path


def fairness_metrics(
    pipe: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.3
) -> dict[str, float]:
    """Disparidade entre faixas etárias na decisão 'alto risco' (proba >= threshold)."""
    age_band = pd.cut(X["age"], bins=[17, 29, 49, 130], labels=["18-29", "30-49", "50+"])
    y_pred = pipe.predict_proba(X)[:, 1] >= threshold
    return {
        "fairness_dem_parity_diff_age": float(
            demographic_parity_difference(y, y_pred, sensitive_features=age_band)
        ),
        "fairness_eq_odds_diff_age": float(
            equalized_odds_difference(y, y_pred, sensitive_features=age_band)
        ),
    }
