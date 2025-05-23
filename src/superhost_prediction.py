import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import List, Tuple


def train_superhost_classifier(X: pd.DataFrame, y: pd.Series, model_type: str = 'xgb'):
    """Train a classifier to predict superhost status."""
    if model_type == 'xgb':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model


def get_feature_importance(model, feature_names: List[str]) -> pd.Series:
    """Return feature importances as a pandas Series."""
    importances = model.feature_importances_
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)
