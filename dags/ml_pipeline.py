import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator


def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data


def preprocess_data(**kwargs):
    
    task_instance = kwargs['ti']
    
    df = task_instance.xcom_pull(task_ids="read_csv")

    le = LabelEncoder()

    df['Sex'] = le.fit_transform(df['Sex'])
    df['RestingECG'] = le.fit_transform(df['RestingECG'])
    df['ChestPainType'] = le.fit_transform(df['ChestPainType'])
    df['ExerciseAngina'] = le.fit_transform(df['ExerciseAngina'])
    df['ST_Slope'] = le.fit_transform(df['ST_Slope'])
    
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train.to_dict(), X_test.to_dict(), y_train.to_dict(), y_test.to_dict()


def train_ada_boost( **kwargs):
    
    task_instance = kwargs['ti']
    
    X_train_dict, _, y_train_dict, _ = task_instance.xcom_pull(task_ids="preprocess_data")
    
    X_train = pd.DataFrame.from_dict(X_train_dict)
    y_train = pd.Series(y_train_dict)
    
    ada_model = AdaBoostClassifier(n_estimators=500, learning_rate=0.01, random_state=0)
    ada_model.fit(X_train, y_train)
    
    model_filepath = f"{model_path}/ada_classifier.pkl"
    joblib.dump(ada_model, model_filepath)
    
    return model_filepath


def test_model(**kwargs):
    
    task_instance = kwargs['ti']
    
    model_filepath = task_instance.xcom_pull(task_ids="train_ada_boost")
    print(model_filepath)

    model = joblib.load(model_filepath)
    
    _, X_test_dict, _, y_test_dict = task_instance.xcom_pull(task_ids="preprocess_data")
    
    X_test = pd.DataFrame.from_dict(X_test_dict)
    y_test = pd.Series(y_test_dict)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("AdaBoost Classifier Model Accuracy:", accuracy_score(y_test, y_pred))
    return accuracy


default_args = {
    "owner": "jino",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "start_date": datetime(2023, 12, 1),
}


dag = DAG(
    "heart_attack_classification",
    default_args=default_args,
    description="A pipeline to read CSV, preprocess data, train and test a simple classification model",
    schedule_interval=timedelta(days=1),
    catchup=False,
)

model_path = "/opt/airflow/model" 
file_path = "/opt/airflow/data/heart.csv"

t1 = PythonOperator(
    task_id="read_csv",
    python_callable=read_csv,
    op_args=[file_path],
    dag=dag,
)

t2 = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_data,
    provide_context=True,
    dag=dag,
)

t3 = PythonOperator(
    task_id="train_ada_boost",
    python_callable=train_ada_boost,
    provide_context=True,
    dag=dag,
)

t4 = PythonOperator(
    task_id="test_model",
    python_callable=test_model,
    provide_context=True,
    dag=dag,
)


t1 >> t2 >> t3 >> t4