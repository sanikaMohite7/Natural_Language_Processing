import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

text = "Natural Language Processing allows machines to understand human language." 

words = word_tokenize(text)

pos_tags = nltk.pos_tag(words)

print("Pos Tags: \n", pos_tags)