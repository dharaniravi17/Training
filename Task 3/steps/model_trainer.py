from zenml import step
from model_dev import LinearRegressionModel, RandomForestModel
from sklearn.model_selection import train_test_split
from typing import Tuple, Any
import pandas as pd

@step
def train_models(df: pd.DataFrame) -> Tuple[Any, Any, pd.DataFrame, pd.Series]:
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_model = LinearRegressionModel().train(X_train, y_train)
    rf_model = RandomForestModel().train(X_train, y_train)

    return lr_model, rf_model, X_test, y_test
