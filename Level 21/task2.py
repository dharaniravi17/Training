import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_encoded = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

def create_model(activation):
    model = Sequential([
        Dense(16, input_shape=(X.shape[1],), activation=activation),
        Dense(3, activation='softmax')
    ])
    return model

activations = ['relu', 'sigmoid', 'tanh']
histories = {}

for act in activations:
    model = create_model(act)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"\nTraining model with {act} activation:")
    history = model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=0)
    histories[act] = history.history['loss']

plt.figure(figsize=(10, 6))
for act in activations:
    plt.plot(histories[act], label=f'{act} activation')
plt.title('Activation Function Comparison - Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend()
plt.grid(True)
plt.show()
