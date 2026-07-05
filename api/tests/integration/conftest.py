"""Setup dos testes de integração: exigem a infra do compose de pé.

Se o MLflow não responder, todos os testes desta pasta são PULADOS (não
falham): permite rodar a suíte unitária em qualquer máquina sem Docker.
"""
import urllib.request

import pytest
from dotenv import load_dotenv

load_dotenv()


def _mlflow_disponivel() -> bool:
    try:
        with urllib.request.urlopen("http://localhost:5000/health", timeout=2) as resp:
            return resp.status == 200
    except OSError:
        return False


def pytest_collection_modifyitems(items):
    if _mlflow_disponivel():
        return
    skip = pytest.mark.skip(reason="infra indisponível: suba com 'docker compose up -d' em infra/")
    for item in items:
        item.add_marker(skip)
