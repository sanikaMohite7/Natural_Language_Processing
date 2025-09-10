import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt
import seaborn as sns

# Sample data (replace with your dataset)
corpus = [
    "Natural language processing enables computers to understand human language",
    "Word embeddings like Word2Vec represent words in vector space",
    "Bag of words and TF-IDF are traditional NLP approaches",
    "Deep learning models improve performance in NLP tasks"
]

# 1. Bag-of-Words (Count Occurrence)

count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(corpus)

print("Vocabulary:", count_vectorizer.get_feature_names_out())
print("\nBag-of-Words (Count Occurrence):")
print(pd.DataFrame(count_matrix.toarray(), columns=count_vectorizer.get_feature_names_out()))

# Visualization: Bag-of-Words
plt.figure(figsize=(8, 4))
sns.heatmap(pd.DataFrame(count_matrix.toarray(), columns=count_vectorizer.get_feature_names_out()), annot=True, cmap="Blues")
plt.title("Bag-of-Words (Count Occurrence)")
plt.xlabel("Words")
plt.ylabel("Documents")
plt.tight_layout()
plt.show()

# 2. Normalized Count Occurrence (Frequency)

normalized_count = count_matrix.toarray().astype(float)
normalized_count = normalized_count / normalized_count.sum(axis=1, keepdims=True)

print("\nNormalized Count Occurrence (Term Frequency):")
print(pd.DataFrame(normalized_count, columns=count_vectorizer.get_feature_names_out()))

# Visualization: Normalized Count Occurrence
plt.figure(figsize=(8, 4))
sns.heatmap(pd.DataFrame(normalized_count, columns=count_vectorizer.get_feature_names_out()), annot=True, cmap="Greens")
plt.title("Normalized Count Occurrence (Term Frequency)")
plt.xlabel("Words")
plt.ylabel("Documents")
plt.tight_layout()
plt.show()

# 3. TF-IDF

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

print("\nTF-IDF Matrix:")
print(pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out()))

# Visualization: TF-IDF
plt.figure(figsize=(8, 4))
sns.heatmap(pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out()), annot=True, cmap="Purples")
plt.title("TF-IDF Matrix")
plt.xlabel("Words")
plt.ylabel("Documents")
plt.tight_layout()
plt.show()


# 4. Word2Vec Embeddings

# Tokenize corpus
tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]

# Train Word2Vec
w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=50, window=5, min_count=1, workers=4)

# Example: Get embedding of a word
word = "language"
print(f"\nWord2Vec embedding for '{word}':\n", w2v_model.wv[word])

# Example: Find most similar words
print("\nMost similar words to 'language':", w2v_model.wv.most_similar("language"))