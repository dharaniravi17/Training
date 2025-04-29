import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

df = pd.read_csv("Tweets.csv")
df = df[df['airline_sentiment'].isin(['positive', 'negative'])]
df['label'] = df['airline_sentiment'].map({'positive': 1, 'negative': 0})

texts = df['text'].values
labels = df['label'].values

vocab_size = 10000
max_length = 50
embedding_dim = 100

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length, name="embedding"),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")
embedding_layer = model.get_layer("embedding")
embedding_weights = embedding_layer.get_weights()[0] 
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
reduced_embeddings = tsne.fit_transform(embedding_weights[:500]) 
words = list(word_index.keys())[:500]

plt.figure(figsize=(15, 10))
for i in range(500):
    plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
    if i % 25 == 0:
        plt.annotate(words[i], (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

plt.title("2D Visualization of Word Embeddings (t-SNE)")
plt.xlabel("TSNE Component 1")
plt.ylabel("TSNE Component 2")
plt.grid(True)
plt.show()
