"""
This is a boilerplate pipeline 'PreparacaoDados'
generated using Kedro 0.19.11
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from pycaret.classification import ClassificationExperiment

def prepare_data(raw_data):
    columns_to_keep = ["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance", "shot_made_flag"]
    df = raw_data[columns_to_keep]
    df = df.dropna()
    scaler = StandardScaler()
    df[["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance"]] = scaler.fit_transform(
        df[["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance"]]
    )
    return df

def split_bases(filtered_data):
    train_df, test_df = train_test_split(filtered_data, test_size=0.2, stratify=filtered_data["shot_made_flag"], random_state=42)
    return train_df, test_df