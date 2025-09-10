import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    "Natural language processing enables computers to understand human language",
    "Word embeddings like Word2Vec represent words in vector space",
    "Bag of words and TF-IDF are traditional NLP approaches",
    "Deep learning models improve performance in NLP tasks"
]

# Bag-of-Words
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(corpus)
bow_df = pd.DataFrame(count_matrix.toarray(), columns=count_vectorizer.get_feature_names_out())

# Normalized Term Frequency
norm_tf = bow_df.div(bow_df.sum(axis=1), axis=0)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

print("Bag-of-Words:\n", bow_df)
print("\nNormalized Term Frequency:\n", norm_tf)
print("\nTF-IDF:\n", tfidf_df)