"""Exceções do domínio — o contrato de erros das ports.

Adapters capturam exceções da sua tecnologia (boto3, psycopg, mlflow...)
e as traduzem para estas na fronteira. O domínio e os entrypoints só
conhecem este vocabulário.
"""


class DomainError(Exception):
    """Base das exceções do domínio."""


class ModelUnavailableError(DomainError):
    """O modelo não pôde ser carregado ou não está pronto para prever."""


class PredictionLogError(DomainError):
    """A predição não pôde ser registrada para auditoria."""


class PredictionNotFoundError(DomainError):
    """Não existe predição registrada com o id informado."""
