import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load and preprocess data
iris = load_iris()
X = iris.data[:100]  # Select only first 100 rows
y = iris.target[:100].reshape(-1, 1)

# Binary classification (Setosa vs Versicolor)
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Split for fair comparison
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# 3. Manually define weights and biases
np.random.seed(42)
weights = np.random.randn(X_train.shape[1], y_train.shape[1])
bias = np.random.randn(y_train.shape[1])

# 4. Forward pass using sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

manual_output = sigmoid(np.dot(X_test, weights) + bias)

# 5. TensorFlow model
model = Sequential([
    Dense(1, input_shape=(X_train.shape[1],), activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train[:, 0], epochs=10, verbose=0)

keras_output = model.predict(X_test).flatten()

# 6. Plotting
plt.figure(figsize=(10, 5))
plt.plot(manual_output, label='Manual Prediction (Sigmoid)', marker='o')
plt.plot(keras_output, label='Keras Model Prediction', marker='x')
plt.plot(y_test[:, 0], label='Actual', linestyle='--', color='black')
plt.title('Manual vs Keras Predictions vs Actual Labels')
plt.xlabel('Sample Index')
plt.ylabel('Predicted Probability')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Display table
print("\n🔹 Sample Predictions:")
comparison_df = pd.DataFrame({
    'Manual_Pred': manual_output.round(2).flatten(),
    'Keras_Pred': keras_output.round(2).flatten(),
    'Actual': y_test[:, 0]
})
print(comparison_df.head(10))
