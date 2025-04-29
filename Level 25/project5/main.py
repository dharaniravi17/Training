import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:\\Users\\Dharani Ravi\\Desktop\\ML projects\\stockprice\\data\\AAPL.csv",  skiprows=2)  
print(df.head())

data = df['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
def create_sequences(data, timesteps=10):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(data[i+timesteps])
    return np.array(X), np.array(y)

timesteps = 10
X, y = create_sequences(data_scaled, timesteps)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, activation='tanh', input_shape=(timesteps, 1)),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse')
history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_test, y_test)
)

y_pred = model.predict(X_test)
y_test_real = scaler.inverse_transform(y_test)
y_pred_real = scaler.inverse_transform(y_pred)
plt.figure(figsize=(10,6))
plt.plot(y_test_real, label='Actual Price')
plt.plot(y_pred_real, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction using Simple RNN')
plt.legend()
plt.show()

mse = tf.keras.losses.MeanSquaredError()
test_mse = mse(y_test, y_pred).numpy()
print(f"\nTest MSE: {test_mse:.4f}")
