from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

# Define the model deployment step using MLflow
@mlflow_model_deployer_step
def deploy_model(model):
    """Deploy the model using MLflow."""
    return model
