import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow

def prepare_data(raw_data):
    columns_to_keep = ["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance", "shot_made_flag"]
    df = raw_data[columns_to_keep]
    df = df.dropna()

    with mlflow.start_run(run_name="Prepare_Data", nested=True):
        scaler = StandardScaler()
        df[["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance"]] = scaler.fit_transform(
            df[["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance"]]
        )

        # Registrar a dimensão do dataset resultante
        mlflow.log_metric("dataset_rows", df.shape[0])
        mlflow.log_metric("dataset_columns", df.shape[1])

    return df


def split_bases(filtered_data):
    test_size = 0.2  # Porcentagem de dados para teste
    random_state = 42  # Para reprodutibilidade

    with mlflow.start_run(run_name="Split_Bases", nested=True):
        train_df, test_df = train_test_split(
            filtered_data, test_size=test_size, stratify=filtered_data["shot_made_flag"], random_state=random_state
        )

        # Registrar parâmetros
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        # Registrar métricas
        mlflow.log_metric("train_size", len(train_df))
        mlflow.log_metric("test_size", len(test_df))

    return train_df, test_df