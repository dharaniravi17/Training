from zenml import pipeline
from steps.dynamic_importer import dynamic_importer
from steps.predictor import predictor
from zenml.integrations.mlflow.steps import mlflow_prediction_service_loader_step

@pipeline
def inference_pipeline():
    data = dynamic_importer()
    service = mlflow_prediction_service_loader_step()
    predictor(service=service, data=data)
