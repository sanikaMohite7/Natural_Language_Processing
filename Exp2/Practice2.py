import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

text = "Machine learning is a subset of artificial intelligence that enables computers to understand and process human language effectively."

words = word_tokenize(text)

pos_tags = nltk.pos_tag(words)

print("Pos Tags: \n" ,pos_tags)