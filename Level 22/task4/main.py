import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

titanic = pd.read_csv("D:\\titanic\\train.csv")

features = ['Pclass', 'Age', 'Fare', 'Sex']
titanic = titanic[features + ['Survived']]
titanic = titanic.dropna()

titanic['Sex'] = LabelEncoder().fit_transform(titanic['Sex'])  
X = titanic[features].values
y = titanic['Survived'].values.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_size = x_train.shape[1] 
hidden_size = 5
output_size = 1
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-8, 1-1e-8)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
learning_rate = 0.01
epochs = 50
loss_history = []

for epoch in range(epochs):
    z1 = np.dot(x_train, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = sigmoid(z2)
    
    loss = binary_cross_entropy(y_train, y_pred)
    loss_history.append(loss)
    
    dz2 = y_pred - y_train 
    dW2 = np.dot(a1.T, dz2) / len(x_train)
    db2 = np.sum(dz2, axis=0, keepdims=True) / len(x_train)

    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * relu_derivative(z1)
    dW1 = np.dot(x_train.T, dz1) / len(x_train)
    db1 = np.sum(dz1, axis=0, keepdims=True) / len(x_train)
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
plt.plot(range(epochs), loss_history)
plt.title('Loss Reduction Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross Entropy Loss')
plt.grid(True)
plt.show()
