from zenml.client import Client
from pipelines.model_deployment_pipeline import model_deployment_pipeline

if __name__ == "__main__":
    # Set the experiment tracker (optional)
    Client().set_experiment_tracker("mlflow_tracker")

    # Run the pipeline
    model_deployment_pipeline()
