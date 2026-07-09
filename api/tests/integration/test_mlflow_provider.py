"""Integração: o adapter contra o MLflow/MinIO reais (infra do compose)."""
import pytest

from credit_risk_api.domain.exceptions import ModelUnavailableError
from credit_risk_api.domain.models import CreditApplication
from credit_risk_api.adapters.mlflow_provider import MLflowModelProvider


def _application(**overrides) -> CreditApplication:
    base = dict(
        revolving_utilization=0.2,
        age=45,
        n_late_30_59=0,
        debt_ratio=0.3,
        monthly_income=5000.0,
        n_open_credit_lines=5,
        n_late_90=0,
        n_real_estate_loans=1,
        n_late_60_89=0,
        n_dependents=1,
    )
    return CreditApplication(**{**base, **overrides})


@pytest.fixture(scope="module")
def provider() -> MLflowModelProvider:
    # scope=module: carrega o champion UMA vez para todos os testes do arquivo
    return MLflowModelProvider()


class TestMLflowModelProvider:

    def test_carrega_o_champion_e_expoe_versao(self, provider):
        # o champion muda conforme o gate promove; o contrato é expor A versão dele
        from mlflow import MlflowClient

        esperado = MlflowClient().get_model_version_by_alias("credit-risk-model", "champion").version
        assert provider.model_version() == esperado

    def test_probabilidade_e_um_float_valido(self, provider):
        proba = provider.predict_probability(_application())
        assert isinstance(proba, float)
        assert 0.0 <= proba <= 1.0

    def test_historico_ruim_aumenta_o_risco(self, provider):
        # sanity check comportamental: atrasos graves devem elevar a probabilidade
        bom_pagador = provider.predict_probability(_application())
        mau_pagador = provider.predict_probability(
            _application(n_late_90=4, n_late_30_59=3, revolving_utilization=0.95)
        )
        assert mau_pagador > bom_pagador

    def test_renda_ausente_nao_quebra(self, provider):
        # o imputer DENTRO do pipeline aplica a mediana do treino
        proba = provider.predict_probability(
            _application(monthly_income=None, n_dependents=None)
        )
        assert 0.0 <= proba <= 1.0


def test_modelo_inexistente_vira_erro_de_dominio():
    with pytest.raises(ModelUnavailableError):
        MLflowModelProvider(model_name="nao-existe")
