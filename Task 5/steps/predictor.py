from zenml import step
import numpy as np
from zenml.integrations.mlflow.services import MLFlowDeploymentService

@step
def predictor(service: MLFlowDeploymentService, data: np.ndarray) -> np.ndarray:
    if not service.is_running:
        service.start(timeout=60)  # ensure service is running

    prediction = service.predict(data)
    print("🔮 Predictions:", prediction)

    return prediction
