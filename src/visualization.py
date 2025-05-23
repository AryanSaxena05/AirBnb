import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_importance(importances: pd.Series, top_n: int = 10, title: str = "Feature Importances"):
    plt.figure(figsize=(10, 6))
    importances.head(top_n).plot(kind='barh', color='steelblue')
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_cluster_summary(cluster_summary: pd.DataFrame, value_col: str, title: str):
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Cluster', y=value_col, data=cluster_summary, palette="viridis")
    plt.title(title)
    plt.xlabel("Cluster")
    plt.ylabel(value_col)
    plt.show()
