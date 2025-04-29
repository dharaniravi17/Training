from zenml import pipeline
from steps.data_loader import load_data
from steps.data_cleaner import clean_data
from steps.model_trainer import train_model
from steps.model_evaluator import evaluate_model

@pipeline
def regression_pipeline():
    df = load_data()
    cleaned_df = clean_data(df)
    model = train_model(cleaned_df)
    evaluate_model(cleaned_df, model)
