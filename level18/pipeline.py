import pandas as pd
from zenml.steps import step
from zenml.pipelines import pipeline
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model
from steps.custom_data_loader import custom_data_loader
from steps.clean_data import clean_data

# Step 1: Define the pipeline
@pipeline
def model_pipeline():
    # Load data, clean data, train model, and evaluate model
    data_loader = custom_data_loader()  # Load raw data
    cleaned_data = clean_data(data_loader)  # Clean the data (could include handling missing values, etc.)
    model = train_model(cleaned_data)  # Train the model with cleaned data
    mse = evaluate_model(model, cleaned_data)  # Evaluate the model (passing cleaned data)
    
    # Return the evaluation result if needed
    return mse

# Step 2: Run the pipeline
if __name__ == "__main__":
    model_pipeline()
