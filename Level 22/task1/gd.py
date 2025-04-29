import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("D:\\bostonh\\BostonHousing.csv")

X = data.drop('medv', axis=1).values  
y = data['medv'].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

np.random.seed(42)
weights = np.random.randn(X_train.shape[1]) * 0.01 
bias = 0.0

print(np.isnan(X).sum(), np.isinf(X).sum())
print(np.isnan(y).sum(), np.isinf(y).sum())

learning_rate = 0.001  
epochs = 100
losses = []

for epoch in range(epochs):
    y_pred = np.dot(X_train, weights) + bias
    
    if np.any(np.isnan(y_pred)):
        print(f"NaN detected in predictions at epoch {epoch}")
        break
    loss = np.mean((y_pred - y_train) ** 2)
    losses.append(loss)
    dw = (2 / X_train.shape[0]) * np.dot(X_train.T, (y_pred - y_train))
    db = (2 / X_train.shape[0]) * np.sum(y_pred - y_train)
    
    weights -= learning_rate * dw
    bias -= learning_rate * db
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss = {loss:.4f}')

plt.figure(figsize=(8,5))
plt.plot(range(len(losses)), losses, marker='o')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.grid()
plt.show()

y_pred_test = np.dot(X_test, weights) + bias
comparison = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_test
})

print("\nComparison of Actual vs Predicted values:\n")
print(comparison.head(10))  
