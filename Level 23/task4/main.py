import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
df = pd.read_csv("D:\\creditcard\\creditcard.csv")  
X = df.drop('Class', axis=1)
y = df['Class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
def create_ann():
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
model_baseline = create_ann()
model_baseline.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

y_pred_baseline = (model_baseline.predict(X_test) > 0.5).astype("int32")
print("Baseline Model Metrics:")
print(classification_report(y_test, y_pred_baseline))
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

model_balanced = create_ann()
model_balanced.fit(X_resampled, y_resampled, epochs=10, batch_size=32, verbose=1)

y_pred_balanced = (model_balanced.predict(X_test) > 0.5).astype("int32")
print("Balanced Model Metrics (SMOTE):")
print(classification_report(y_test, y_pred_balanced))
from sklearn.metrics import precision_score, recall_score, f1_score
precision_baseline = precision_score(y_test, y_pred_baseline)
recall_baseline = recall_score(y_test, y_pred_baseline)
f1_baseline = f1_score(y_test, y_pred_baseline)
precision_balanced = precision_score(y_test, y_pred_balanced)
recall_balanced = recall_score(y_test, y_pred_balanced)
f1_balanced = f1_score(y_test, y_pred_balanced)

comparison_table = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score'],
    'Before Balancing': [precision_baseline, recall_baseline, f1_baseline],
    'After Balancing (SMOTE)': [precision_balanced, recall_balanced, f1_balanced]
})

print("\nComparison Table:")
print(comparison_table.to_string(index=False))
