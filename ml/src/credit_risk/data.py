from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "cs-training.csv"

COLS_ATRASO = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate',
]


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Regras de limpeza de domínio (constantes fixas, sem estatística aprendida -> sem leakage)."""
    df = df[df['age'] >= 18].copy()
    # 96/98 são códigos sentinela; o máximo real de atrasos em 2 anos fica abaixo de 20
    for c in COLS_ATRASO:
        df[c] = df[c].clip(upper=20)
    # Utilização é proporção; >10 (1000% do limite) é erro de dado. Cap conservador.
    df['RevolvingUtilizationOfUnsecuredLines'] = df['RevolvingUtilizationOfUnsecuredLines'].clip(upper=10)
    return df
