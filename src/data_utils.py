import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates, handle missing values."""
    df = df.drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))
    return df


def scale_features(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale selected features using StandardScaler."""
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    return df_scaled, scaler
