from zenml import pipeline
from steps.data_loader import load_data
from steps.model_trainer import train_models
from steps.model_evaluator import evaluate_models

@pipeline
def comparison_pipeline():
    df = load_data()
    lr_model, rf_model, X_test, y_test = train_models(df)
    evaluate_models(lr_model, rf_model, X_test, y_test)
