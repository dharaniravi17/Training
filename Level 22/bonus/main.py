import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target, columns=["MEDV"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train).flatten()
y_test = np.array(y_test).flatten()
def predict(X, weights, bias):
    return np.dot(X, weights) + bias

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(X, y, learning_rate=0.01, epochs=100):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    loss_history = []

    for epoch in range(epochs):
        y_pred = predict(X, weights, bias)
        error = y_pred - y

        dw = (2/n_samples) * np.dot(X.T, error)
        db = (2/n_samples) * np.sum(error)

        weights -= learning_rate * dw
        bias -= learning_rate * db

        loss = mse(y, y_pred)
        loss_history.append(loss)

    return weights, bias, loss_history

def gradient_descent_momentum(X, y, learning_rate=0.01, epochs=100, beta=0.9):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    v_w = np.zeros(n_features)  
    v_b = 0                     
    loss_history = []

    for epoch in range(epochs):
        y_pred = predict(X, weights, bias)
        error = y_pred - y
        dw = (2/n_samples) * np.dot(X.T, error)
        db = (2/n_samples) * np.sum(error)
        v_w = beta * v_w + (1 - beta) * dw
        v_b = beta * v_b + (1 - beta) * db

        weights -= learning_rate * v_w
        bias -= learning_rate * v_b

        loss = mse(y, y_pred)
        loss_history.append(loss)

    return weights, bias, loss_history

_, _, loss_gd = gradient_descent(X_train, y_train, learning_rate=0.01, epochs=100)
_, _, loss_momentum = gradient_descent_momentum(X_train, y_train, learning_rate=0.01, epochs=100)

plt.figure(figsize=(10,6))
plt.plot(loss_gd, label='Standard Gradient Descent')
plt.plot(loss_momentum, label='Gradient Descent with Momentum')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss Curve Comparison')
plt.legend()
plt.grid(True)
plt.show()
