import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import pandas as pd

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

def create_model():
    model = Sequential([
        Dense(256, activation='relu', input_shape=(784,)),
        Dense(256, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

cpu_times = None
cpu_acc = None

with tf.device('/CPU:0'):
    model_cpu = create_model()
    start_time = time.time()
    model_cpu.fit(x_train, y_train, epochs=10, verbose=2)
    cpu_times = time.time() - start_time
    _, cpu_acc = model_cpu.evaluate(x_test, y_test, verbose=0)

print(f"CPU training time: {cpu_times:.2f} seconds")
print(f"CPU test accuracy: {cpu_acc:.4f}")

gpu_times = None
gpu_acc = None
model_gpu = create_model()
start_time = time.time()
model_gpu.fit(x_train, y_train, epochs=10, verbose=2)
gpu_times = time.time() - start_time
_, gpu_acc = model_gpu.evaluate(x_test, y_test, verbose=0)

print(f"GPU training time: {gpu_times:.2f} seconds")
print(f"GPU test accuracy: {gpu_acc:.4f}")

results = pd.DataFrame({
    'Device': ['CPU', 'GPU'],
    'Training Time (seconds)': [cpu_times, gpu_times],
    'Test Accuracy': [cpu_acc, gpu_acc]
})

print("\nComparison Table:")
print(results)
