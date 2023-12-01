# Dockerfile
FROM apache/airflow:2.7.3

USER airflow

# Install additional dependencies
RUN pip install --no-cache-dir joblib scikit-learn

USER airflow
