"""Contrato de qualidade dos dados (pandera).

Valida o DataFrame APÓS clean() + add_features(): se o gate falhar,
o pipeline de treino para antes de produzir um modelo com dado podre.
"""
import pandera.pandas as pa

from credit_risk.data import COLS_ATRASO

TARGET = "SeriousDlqin2yrs"

schema = pa.DataFrameSchema(
    {
        TARGET: pa.Column(int, pa.Check.isin([0, 1])),
        "RevolvingUtilizationOfUnsecuredLines": pa.Column(float, pa.Check.in_range(0, 10)),
        "age": pa.Column(int, pa.Check.in_range(18, 130)),
        "DebtRatio": pa.Column(float, pa.Check.ge(0)),
        "MonthlyIncome": pa.Column(float, pa.Check.ge(0), nullable=True),
        "NumberOfOpenCreditLinesAndLoans": pa.Column(int, pa.Check.ge(0)),
        "NumberRealEstateLoansOrLines": pa.Column(int, pa.Check.ge(0)),
        "NumberOfDependents": pa.Column(float, pa.Check.ge(0), nullable=True),
        "declarou_renda": pa.Column(int, pa.Check.isin([0, 1])),
        **{c: pa.Column(int, pa.Check.in_range(0, 20)) for c in COLS_ATRASO},
    },
    strict=True,   # nenhuma coluna inesperada
    coerce=False,
)
