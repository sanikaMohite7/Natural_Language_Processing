import nltk
from nltk.tokenize import word_tokenize

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