import numpy as np

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Binary cross-entropy loss function
def binary_cross_entropy(y_true, y_pred):
    m = len(y_true)
    return -1/m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Initialize weights and bias
def initialize_parameters(input_size):
    W = np.zeros((input_size, 1))
    b = 0
    return W, b

# Forward pass
def forward_propagation(X, W, b):
    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    return A

# Backward pass (gradient descent)
def backward_propagation(X, y, A):
    m = X.shape[0]
    dZ = A - y.reshape(-1, 1)
    dW = 1/m * np.dot(X.T, dZ)
    db = 1/m * np.sum(dZ)
    return dW, db

# Gradient descent step
def gradient_descent(X, y, W, b, learning_rate, num_iterations):
    for i in range(num_iterations):
        A = forward_propagation(X, W, b)
        loss = binary_cross_entropy(y, A)
        
        # Print loss every 10 iterations
        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {loss}")
        
        # Backpropagation
        dW, db = backward_propagation(X, y, A)
        
        # Update parameters
        W -= learning_rate * dW
        b -= learning_rate * db
        
    return W, b
