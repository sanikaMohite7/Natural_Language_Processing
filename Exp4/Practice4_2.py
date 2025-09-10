import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

text = """
Natural Language Processing is an exciting field that deals with the interaction 
between computers and humans using natural language. It involves many tasks such as 
tokenization, stemming, and removing stop words to process and analyze large amounts 
of natural language data.
"""

# Process the text
words = word_tokenize(text.lower())
stop_words = set(stopwords.words('english'))
stop_words_list = [word for word in words if word in stop_words and word.isalnum()]
stop_words_count = len(stop_words_list)

ps = PorterStemmer()
stemmed_words = [ps.stem(word) for word in words if word.isalnum() and word not in stop_words]
stemmed_words_count = len(stemmed_words)

# Count word frequencies
stop_words_freq = Counter(stop_words_list)
stemmed_words_freq = Counter(stemmed_words)

# Create figure with two subplots
plt.figure(figsize=(15, 6))

# Plot 1: Stop Words
plt.subplot(1, 2, 1)
if stop_words_freq:
    plt.bar(stop_words_freq.keys(), stop_words_freq.values(), color='red')
    plt.title(f'Stop Words (Total: {stop_words_count})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

# Plot 2: Stemmed Words
plt.subplot(1, 2, 2)
if stemmed_words_freq:
    plt.bar(stemmed_words_freq.keys(), stemmed_words_freq.values(), color='green')
    plt.title(f'Stemmed Words (Total: {stemmed_words_count})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

# Print counts
print(f"Original text: {text}")
print(f"\nNumber of stop words: {stop_words_count}")
print(f"Number of stemmed words: {stemmed_words_count}")
print("\nList of stemmed words:", stemmed_words)

plt.tight_layout()
plt.show()