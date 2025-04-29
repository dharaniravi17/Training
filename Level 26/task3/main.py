import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

texts = df['text'].values
labels = df['label'].values

vocab_size = 5000
max_length = 50

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length),
    Bidirectional(LSTM(32)),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int).flatten()

acc = accuracy_score(y_test, y_pred_labels)
print(f"\nTest Accuracy: {acc:.4f}")
for i in range(5):
    print(f"\nText: {texts[i]}")
    print(f"True Label: {labels[i]}, Predicted Label: {int(y_pred_labels[i])}")

