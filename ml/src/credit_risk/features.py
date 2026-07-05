"""Criação de features de negócio.

Somente regras fixas (sem estatística aprendida nos dados), por isso podem
rodar antes do split sem risco de leakage. Transformações que aprendem
parâmetros (imputação, escala) pertencem ao Pipeline em train.py.
"""
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features derivadas ao DataFrame (sem mutar o original)."""
    df = df.copy()
    # Missing de renda é informativo: quem não declara tem perfil de risco distinto.
    # A flag precisa existir ANTES da imputação, senão a informação se perde.
    df['declarou_renda'] = df['MonthlyIncome'].notna().astype(int)
    return df
