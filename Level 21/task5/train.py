import numpy as np
from data_preprocessing import preprocess_data
from neuron import initialize_parameters, gradient_descent, forward_propagation

# Load and preprocess data
X_train, X_test, y_train, y_test = preprocess_data()

# Initialize parameters
input_size = X_train.shape[1]  # Number of features
W, b = initialize_parameters(input_size)

# Train the model
learning_rate = 0.01
num_iterations = 100
W, b = gradient_descent(X_train, y_train, W, b, learning_rate, num_iterations)

# Make predictions on the test set
A_test = forward_propagation(X_test, W, b)
y_pred = (A_test >= 0.5).astype(int)

# Compute accuracy
accuracy = np.mean(y_pred.flatten() == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
