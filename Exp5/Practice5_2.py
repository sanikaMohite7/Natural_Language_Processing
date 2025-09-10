import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

# Sample corpus (replace with your dataset)
corpus = [
    "The king and queen ruled the kingdom.",
    "A man and a woman are walking in the park.",
    "The computer is an amazing invention.",
    "Queen Elizabeth is a famous monarch.",
    "A programmer uses a computer to solve problems.",
    "The king greeted the people.",
    "Women and men have equal rights.",
    "The queen and the king attended the ceremony."
]

# Tokenize sentences
tokenized_corpus = [sentence.lower().split() for sentence in corpus]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=2, seed=42)

# Chosen words
words = ["king", "queen", "man", "woman", "computer"]

# 1. Vector embeddings
embeddings = {word: model.wv[word] for word in words}

# 2. Most similar words
similar_words = {word: model.wv.most_similar(word, topn=5) for word in words}

# 3. Vector arithmetic: king - man + woman ≈ ?
result_word, similarity = model.wv.most_similar(positive=['king', 'woman'], negative=['man'])[0]

# 4. Present results
print("Word Embeddings (first 5 values):")
for word in words:
    print(f"{word}: {embeddings[word][:5]}")

print("\nMost Similar Words:")
for word in words:
    print(f"{word}: {[w for w, s in similar_words[word]]}")

print(f"\nVector Arithmetic (king - man + woman): {result_word} (similarity: {similarity:.4f})")

# Tabular presentation
sim_df = pd.DataFrame({word: [w for w, _ in similar_words[word]] for word in words})
print("\nMost Similar Words Table:\n", sim_df)

# Optional: Graphical presentation (bar chart for similarity scores of 'king')
king_sim = similar_words["king"]
plt.figure(figsize=(8,4))
plt.bar([w for w, _ in king_sim], [s for _, s in king_sim], color='skyblue')
plt.title("Top 5 Similar Words to 'king'")
plt.ylabel("Similarity Score")
plt.show()