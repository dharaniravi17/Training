# pipeline.py

from zenml import pipeline, step
from model_dev import LinearRegressionModel
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

@step
def load_data():
    data = load_diabetes()
    return train_test_split(data.data, data.target, test_size=0.2)

@step
def train_model(X_train, X_test, y_train, y_test):
    model = LinearRegressionModel()
    trained_model = model.train(X_train, y_train)
    return trained_model

@pipeline
def training_pipeline(load_data_step, train_model_step):
    X_train, X_test, y_train, y_test = load_data_step()
    train_model_step(X_train, X_test, y_train, y_test)
