import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("D:\\churn\\WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.drop('customerID', axis=1, inplace=True)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df.fillna(0, inplace=True)
binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_columns:
    df[col] = df[col].map({'No': 0, 'Yes': 1, 'Female': 0, 'Male': 1})

df = pd.get_dummies(df, columns=['MultipleLines', 'InternetService', 'OnlineSecurity', 
                                 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                                 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'],
                    drop_first=True)

get
X = df.drop('Churn', axis=1).values
y = df['Churn'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model_no_dropout = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_with_dropout = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model_no_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_with_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_no_dropout = model_no_dropout.fit(X_train, y_train, epochs=10, batch_size=32,
                                          validation_data=(X_test, y_test), verbose=1)

history_with_dropout = model_with_dropout.fit(X_train, y_train, epochs=10, batch_size=32,
                                              validation_data=(X_test, y_test), verbose=1)

test_loss_no_dropout, test_acc_no_dropout = model_no_dropout.evaluate(X_test, y_test, verbose=0)
test_loss_with_dropout, test_acc_with_dropout = model_with_dropout.evaluate(X_test, y_test, verbose=0)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history_no_dropout.history['loss'], label='Train Loss (No Dropout)')
plt.plot(history_no_dropout.history['val_loss'], label='Val Loss (No Dropout)')
plt.plot(history_with_dropout.history['loss'], label='Train Loss (With Dropout)')
plt.plot(history_with_dropout.history['val_loss'], label='Val Loss (With Dropout)')
plt.title('Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_no_dropout.history['accuracy'], label='Train Accuracy (No Dropout)')
plt.plot(history_no_dropout.history['val_accuracy'], label='Val Accuracy (No Dropout)')
plt.plot(history_with_dropout.history['accuracy'], label='Train Accuracy (With Dropout)')
plt.plot(history_with_dropout.history['val_accuracy'], label='Val Accuracy (With Dropout)')
plt.title('Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
print("\nComparison of Test Performance:")
print(f"Test Accuracy (No Dropout): {test_acc_no_dropout * 100:.2f}%")
print(f"Test Loss (No Dropout): {test_loss_no_dropout:.4f}")
print(f"Test Accuracy (With Dropout): {test_acc_with_dropout * 100:.2f}%")
print(f"Test Loss (With Dropout): {test_loss_with_dropout:.4f}")
