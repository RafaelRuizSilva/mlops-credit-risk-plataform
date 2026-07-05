-- Roda apenas na PRIMEIRA inicialização do volume (docker-entrypoint-initdb.d).
-- Em volumes já existentes, criar na mão: psql -U mlflow -d mlflow -c "CREATE DATABASE credit_risk"
CREATE DATABASE credit_risk;
