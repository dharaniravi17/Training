# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Step 1: Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Step 2: Preprocess the data
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Step 3: Define a function to build the model
def build_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Train with different batch sizes
batch_sizes = [len(x_train), 1, 32]
histories = {}

for batch_size in batch_sizes:
    print(f"\nTraining with batch size: {batch_size}")
    model = build_model()
    history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_data=(x_test, y_test), verbose=2)
    histories[batch_size] = history

# Step 5: Plot Accuracy and Loss for each
plt.figure(figsize=(16,6))

# Accuracy plot
plt.subplot(1,2,1)
for batch_size, history in histories.items():
    plt.plot(history.history['val_accuracy'], label=f'Batch {batch_size}')
plt.title('Validation Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()

# Loss plot
plt.subplot(1,2,2)
for batch_size, history in histories.items():
    plt.plot(history.history['val_loss'], label=f'Batch {batch_size}')
plt.title('Validation Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()
