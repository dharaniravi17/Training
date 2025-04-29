# steps/model_trainer.py
from zenml import step
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split

@step
def train_model(df: pd.DataFrame) -> LinearRegression:
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
