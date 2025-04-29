from zenml import step
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import mlflow

@step
def evaluate_model(df: pd.DataFrame, model: LinearRegression) -> None:
    X = df.drop("target", axis=1)
    y = df["target"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("mean_squared_error", mse)
    mlflow.log_metric("r2_score", r2)

    print(f"📉 MSE: {mse}")
    print(f"📈 R² Score: {r2}")
