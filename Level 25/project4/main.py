import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard

# Step 1: Load the dataset
df = pd.read_csv("D:\\creditcard\\creditcard.csv")
print("Class distribution:\n", df['Class'].value_counts())

# Step 2: Split features and labels
X = df.drop('Class', axis=1)
y = df['Class']

# Step 3: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Build evaluation function
def evaluate_model(model, X, y, label):
    y_pred = (model.predict(X) > 0.5).astype("int32")
    print(f"\n--- {label} ---")
    print(classification_report(y, y_pred, digits=4))

# Step 6: Build & train model on imbalanced data
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

log_dir_imbal = "logs/imbalanced/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb_imbal = TensorBoard(log_dir=log_dir_imbal)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=2048,
    validation_split=0.2,
    callbacks=[tensorboard_cb_imbal]
)

# Step 7: Evaluate before balancing
evaluate_model(model, X_test, y_test, label="Before Balancing")

# Step 8: Balance the data using SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
print("\nResampled class distribution:", np.bincount(y_resampled))

# Step 9: Build & train model on balanced data
model_bal = Sequential([
    Dense(128, activation='relu', input_shape=(X_resampled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_bal.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

log_dir_bal = "logs/balanced/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb_bal = TensorBoard(log_dir=log_dir_bal)

history_bal = model_bal.fit(
    X_resampled, y_resampled,
    epochs=10,
    batch_size=2048,
    validation_split=0.2,
    callbacks=[tensorboard_cb_bal]
)

# Step 10: Evaluate after balancing
evaluate_model(model_bal, X_test, y_test, label="After Balancing")
