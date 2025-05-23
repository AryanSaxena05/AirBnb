import pandas as pd
from sklearn.cluster import KMeans
from typing import List, Tuple


def run_kmeans(df: pd.DataFrame, features: List[str], n_clusters: int = 6, random_state: int = 42) -> Tuple[pd.DataFrame, KMeans]:
    """Apply KMeans clustering and return dataframe with cluster labels."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(df[features])
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = clusters
    return df_with_clusters, kmeans
