# steps/model_evaluator.py
from zenml import step
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

@step
def evaluate_model(df: pd.DataFrame, model: LinearRegression) -> None:
    X = df.drop("target", axis=1)
    y = df["target"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"📉 Mean Squared Error: {mse}")
    print(f"📈 R2 Score: {r2}")
