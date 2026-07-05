# Imagem oficial do MLflow nao inclui drivers de Postgres nem S3
FROM ghcr.io/mlflow/mlflow:v3.14.0
RUN pip install --no-cache-dir psycopg2-binary boto3
