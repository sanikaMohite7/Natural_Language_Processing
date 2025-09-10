import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

text = "Machine learning algorithms enable computers to analyze large datasets and identify patterns."

words = word_tokenize(text)

pos_tags = nltk.pos_tag(words)

print("POS Tags: \n", pos_tags)

pos_counts = {}
for word, pos in pos_tags:
    pos_counts[pos] = pos_counts.get(pos, 0) + 1

print("POS Frequency: \n", pos_counts)

pos_labels = list(pos_counts.keys())
pos_frequencies = list(pos_counts.values())

plt.figure(figsize=(10, 6))
plt.bar(pos_labels, pos_frequencies, color='skyblue', edgecolor='navy')
plt.title('Parts of Speech Frequency Distribution')
plt.xlabel('POS Tags')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(pos_frequencies):
    plt.text(i, v + 0.05, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()