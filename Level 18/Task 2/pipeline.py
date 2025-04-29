from zenml import pipeline, step
from sklearn import datasets
import numpy as np

@step
def load_digits_data():
    """Step that loads the digits dataset from scikit-learn."""
    # Load the digits dataset
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    
    print(f"Dataset loaded: {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Target classes: {np.unique(y)}")
    
    return X, y

@pipeline
def simple_data_pipeline():
    """A simple pipeline that loads the digits dataset."""
    return load_digits_data()

if __name__ == "__main__":
    # Run the pipeline
    print("Starting ZenML pipeline execution...")
    # In newer ZenML versions, calling the pipeline directly runs it
    pipeline_run = simple_data_pipeline()
    print("Pipeline execution completed!")
    
    # You can access information about the run if needed
    print(f"Pipeline run ID: {pipeline_run.id}")
