# Machine Learning Pipeline using Apache Airflow

![Airflow Logo](https://github.com/apache/airflow/blob/main/airflow/www/static/pin.svg)



This repository contains a simple Apache Airflow pipeline designed to process data, train a machine learning model, and test it using Airflow orchestration. The pipeline is Dockerized for easy deployment and reproducibility.

## Prerequisites

Before getting started, make sure you have Docker installed on your machine.

- [Install Docker Desktop](https://www.docker.com/products/docker-desktop/)

## Quick Start

To run the machine learning pipeline, follow these steps:

1. Start the Airflow service:

    ```bash
    airflow compose up --build
    ```

2. Access the Airflow UI at [http://localhost:8080/](http://localhost:8080/).

3. Log in using the following credentials:
   - Username: airflow
   - Password: airflow

4. Trigger your desired pipeline from the Airflow dashboard.

## Stopping the Service

To stop the Airflow service and remove the containers, run:

```bash
airflow compose down





## Project Structure

machine-learning-pipeline/
│
├── dags/            # Airflow DAGs (Directed Acyclic Graphs)
│   ├── your_dag.py  # Your specific DAG definition
│
├── data/            # Data directory
│
├── models/          # Directory for storing trained ML models
│
└── README.md        # Project README
