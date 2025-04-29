import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_data
from neuron import forward_propagation

# Load and preprocess data
X_train, X_test, y_train, y_test = preprocess_data()

# Use only two features (e.g., 'Pclass' and 'Fare') for visualization
X_train_2d = X_train[:, [0, 2]]  # 'Pclass' and 'Fare'

# Create grid for contour plot
xx, yy = np.meshgrid(np.linspace(X_train_2d[:, 0].min(), X_train_2d[:, 0].max(), 100),
                     np.linspace(X_train_2d[:, 1].min(), X_train_2d[:, 1].max(), 100))

# Flatten grid and make predictions
grid_points = np.c_[xx.ravel(), yy.ravel()]
A_grid = forward_propagation(grid_points, W, b)
Z = (A_grid >= 0.5).reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, edgecolors='k', marker='o')
plt.xlabel('Pclass')
plt.ylabel('Fare')
plt.title('Decision Boundary')
plt.show()
