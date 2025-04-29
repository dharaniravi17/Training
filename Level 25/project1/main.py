# main.py

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Preprocess data
x_train = x_train / 255.0  # normalize pixel values
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28*28)  # flatten into 784 features
x_test = x_test.reshape(-1, 28*28)

y_train = to_categorical(y_train, 10)  # one-hot encoding
y_test = to_categorical(y_test, 10)

# 3. Build Sequential Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),  # first hidden layer
    Dense(64, activation='relu'),                      # second hidden layer
    Dense(10, activation='softmax')                    # output layer
])

# 4. Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(),  # stochastic gradient descent
    metrics=['accuracy']
)

# 5. Train the model
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# 6. Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 7. Predict on test data
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Display 5-10 sample predictions
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    img = x_test[i].reshape(28,28)
    plt.imshow(img, cmap='gray')
    plt.title(f"Pred: {predicted_classes[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 8. Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
