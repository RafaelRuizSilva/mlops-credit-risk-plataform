"""Pipeline de treino: validate -> train -> gate de promoção do champion.

Roda local (Prefect sem servidor):
    uv run python flows/training_flow.py

O gate é a best practice central: um modelo novo só vira champion se
superar o atual na métrica de referência (AUC de teste). Sem gate,
retreinar automaticamente é roleta-russa.
"""
import logging
import sys

from dotenv import load_dotenv
from mlflow import MlflowClient
from prefect import flow, task, get_run_logger

from credit_risk.data import clean, load_raw
from credit_risk.features import add_features
from credit_risk.train import TrainConfig, run_experiment
from credit_risk.validation import schema

MODEL_NAME = "credit-risk-model"
ALIAS = "champion"
GATE_METRIC = "auc_test"


@task
def validate_data() -> int:
    """Gate de qualidade: pandera valida o dataset pós-limpeza."""
    df = add_features(clean(load_raw()))
    schema.validate(df, lazy=True)  # lazy: reporta TODAS as violações, não só a 1ª
    return len(df)


@task
def train_and_register(config: TrainConfig) -> dict:
    return run_experiment(config, register=True)


@task
def promote_if_better(candidate: dict) -> str:
    """Move o alias champion se o candidato superar o campeão atual no gate."""
    logger = get_run_logger()
    client = MlflowClient()
    new_score = candidate["metrics"][GATE_METRIC]

    try:
        current = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        current_score = client.get_run(current.run_id).data.metrics[GATE_METRIC]
    except Exception:
        current, current_score = None, float("-inf")

    if new_score > current_score:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, candidate["registered_version"])
        decision = (
            f"PROMOVIDO: v{candidate['registered_version']} "
            f"({GATE_METRIC} {new_score:.4f} > {current_score:.4f})"
        )
    else:
        decision = (
            f"REJEITADO: v{candidate['registered_version']} "
            f"({GATE_METRIC} {new_score:.4f} <= champion {current_score:.4f} "
            f"da v{current.version})"
        )
    logger.info(decision)
    return decision


@flow(name="credit-risk-training")
def training_pipeline(config: TrainConfig = TrainConfig()) -> str:
    n_rows = validate_data()
    get_run_logger().info("Dados validados: %s linhas", n_rows)
    candidate = train_and_register(config)
    return promote_if_better(candidate)


if __name__ == "__main__":
    import argparse

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO)
    load_dotenv()

    parser = argparse.ArgumentParser(description="Pipeline de treino com gate de promoção")
    parser.add_argument("--model", choices=["logistic", "lightgbm"], default="logistic")
    args = parser.parse_args()
    print(training_pipeline(TrainConfig(model=args.model)))
