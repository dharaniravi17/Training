import pandas as pd  # Import pandas as pd
from sklearn.metrics import mean_squared_error
from zenml.steps import step

# Step 1: Define evaluate_model function as a ZenML step
@step
def evaluate_model(model, test_data: pd.DataFrame) -> float:
    # Step 2: Separate features (X_test) and target (y_test)
    X_test = test_data.drop(columns=["Age"])  # Replace "target" with your actual target column name
    y_test = test_data["Age"]  # Replace "target" with your actual target column name

    # Step 3: Predict the outputs using the trained model
    y_pred = model.predict(X_test)

    # Step 4: Calculate the Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)

    # Step 5: Return the MSE value
    return mse
