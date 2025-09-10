from tkinter.constants import N
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

article_paragraph = """
NASA's Perseverance rover has discovered evidence of ancient lake sediments at the base of Mars' 
Jezero Crater, offering new insights into the planet's watery past. The findings, published in 
Science Advances, suggest that the crater once held a lake that was filled by a small river 
approximately 3.7 billion years ago. The research team, led by scientists from the University 
of California, Los Angeles, analyzed data from the rover's ground-penetrating radar.
"""

tokens = word_tokenize(article_paragraph)
print("\nTokens:")
print(tokens)

pos_tags = pos_tag(tokens)
print("\nPOS Tags:")
print(pos_tags)

print("\nNamed Entities:")
ne_tree = ne_chunk(pos_tags)
print(ne_tree)
