import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

housing = fetch_california_housing()
X = housing.data
y = housing.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def weighted_mse(y_true, y_pred):
    error = tf.square(y_true - y_pred)
    weights = tf.where(error > 1.0, 2.0, 1.0)
    return tf.reduce_mean(error * weights)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(1)
])

model.compile(optimizer='adam', loss=weighted_mse)

history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

y_pred = model.predict(X_test).flatten()


standard_mse = mean_squared_error(y_test, y_pred)

final_custom_loss = weighted_mse(
    tf.constant(y_test, dtype=tf.float32),
    tf.constant(y_pred, dtype=tf.float32)
).numpy()


print("\n🔹 Loss Comparison Table")
print("-------------------------------")
print(f"{'Loss Type':<20} | {'Value'}")
print(f"{'-'*20} | {'-'*10}")
print(f"{'Standard MSE':<20} | {standard_mse:.4f}")
print(f"{'Custom Weighted MSE':<20} | {final_custom_loss:.4f}")
