from zenml import pipeline
from steps.deployment_trigger import deployment_trigger
from steps.model_trainer import train_model
from steps.model_evaluator import evaluate_model
from steps.model_deployer import deploy_model
from sklearn.datasets import load_diabetes
import pandas as pd

@pipeline
def model_deployment_pipeline():
    # Data ingestion
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Train the model
    model = train_model(df)

    # Evaluate the model
    r2 = evaluate_model(model, df)

    # Trigger deployment if R2 > 0.7
    should_deploy = deployment_trigger(r2)

    # Deploy model if trigger is true
    if should_deploy:
        deploy_model(model)
    else:
        print("Model deployment skipped.")
