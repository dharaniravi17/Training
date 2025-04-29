from zenml import step
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import mlflow.sklearn

@step
def train_model(df: pd.DataFrame) -> LinearRegression:
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.sklearn.autolog()
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
