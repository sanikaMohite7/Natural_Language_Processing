from tkinter.constants import N
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

nltk.download('punkt_tab')
nltk.download('averaged_perception_tagger')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

text = "Steve Jobs founded Apple Inc. in California. He was born in San Francisco."

tokens = word_tokenize(text)
print(tokens)

pos_tags = pos_tag(tokens)
print(pos_tags)

ne_tree = ne_chunk(pos_tags)
print(ne_tree)
