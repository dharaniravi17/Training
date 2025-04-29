# model_dev.py

from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Abstract Base Class
class Model(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

# Concrete Implementation of Model
class LinearRegressionModel(Model):
    def train(self, X_train, y_train):
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        return reg

# For quick testing
if __name__ == "__main__":
    # Load sample data
    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2)

    # Instantiate and train model
    model = LinearRegressionModel()
    trained_model = model.train(X_train, y_train)
    print("Model trained successfully:", trained_model)
