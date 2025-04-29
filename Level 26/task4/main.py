import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

df = pd.read_csv("Fake.csv") 
texts = df['text'].dropna().tolist()[:10000] 
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english') and len(word) > 2]
    return tokens

corpus = [clean_text(doc) for doc in texts]
model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=5, workers=4)
key_terms = ["data", "science", "technology", "government", "market"]
for term in key_terms:
    if term in model.wv:
        print(f"\nTop 5 words similar to '{term}':")
        for word, similarity in model.wv.most_similar(term, topn=5):
            print(f"  {word} ({similarity:.2f})")
    else:
        print(f"'{term}' not in vocabulary.")

words = list(model.wv.index_to_key)[:200] 
word_vectors = [model.wv[word] for word in words]
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
word_vec_2d = tsne.fit_transform(word_vectors)
plt.figure(figsize=(14, 10))
plt.scatter(word_vec_2d[:, 0], word_vec_2d[:, 1], alpha=0.7)

for i, word in enumerate(words):
    plt.annotate(word, (word_vec_2d[i, 0], word_vec_2d[i, 1]), fontsize=9)

plt.title("2D Word Embeddings (t-SNE Projection)", fontsize=16)
plt.grid(True)
plt.show()

