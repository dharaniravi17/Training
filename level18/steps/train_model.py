import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from zenml.steps import step

@step
def train_model(df: pd.DataFrame) -> LinearRegression:
    # Handle categorical columns (e.g., 'Gender') in the training step
    if 'Genre' in df.columns:
        le = LabelEncoder()
        df['Genre'] = le.fit_transform(df['Genre'])

    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()  # Drop rows with NaN values

    # Split the data into features and target
    X = df.drop(columns='Age')  # Replace 'target' with the actual target column
    y = df['Age']  # Replace 'target' with the actual target column

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    return model
