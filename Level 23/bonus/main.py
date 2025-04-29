import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE

data = pd.read_csv("D:\\creditcard\\creditcard.csv")
X = data.drop('Class', axis=1)
y = data['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

class WeightedF1Score(tf.keras.metrics.Metric):
    def __init__(self, name='weighted_f1_score', **kwargs):
        super(WeightedF1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
        self.weights = {0: 0.2, 1: 0.8}  

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)
        
        for class_id, weight in self.weights.items():
            true_pos = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, class_id), tf.equal(y_pred, class_id)), self.dtype))
            false_pos = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(y_true, class_id), tf.equal(y_pred, class_id)), self.dtype))
            false_neg = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, class_id), tf.not_equal(y_pred, class_id)), self.dtype))

            self.true_positives.assign_add(weight * true_pos)
            self.false_positives.assign_add(weight * false_pos)
            self.false_negatives.assign_add(weight * false_neg)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train_resampled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy', WeightedF1Score()])

history = model.fit(X_train_resampled, y_train_resampled, epochs=10, batch_size=128, validation_split=0.2, verbose=1)

y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
report = classification_report(y_test, y_pred_classes, output_dict=True)
standard_f1 = report['weighted avg']['f1-score']
custom_weighted_f1 = f1_score(y_test, y_pred_classes, average='weighted', sample_weight=[0.2 if i == 0 else 0.8 for i in y_test])
results_table = pd.DataFrame({
    'Metric': ['Standard Weighted F1', 'Custom Weighted F1'],
    'Score': [standard_f1, custom_weighted_f1]
})

print("\n🔹 Comparison of Metrics:")
print(results_table)

# 7. Brief discussion
print("\n Discussion:")
print("The custom weighted F1 score gives higher importance to correctly classifying the minority class (fraud).")
print("Useful when the cost of missing minority cases (e.g., fraud) is much higher than misclassifying majority cases.")
print("Shows better real-world performance for critical applications like fraud detection.")
