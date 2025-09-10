import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

text = """
NASA's Perseverance rover has discovered evidence of ancient lake sediments at the base of Mars' 
Jezero Crater. The research team from the University of California analyzed the data.
"""

# Process the text
tokens = word_tokenize(text)
tags = pos_tag(tokens)
entities = ne_chunk(tags)

# Count entities
counts = {}
for chunk in entities:
    if hasattr(chunk, 'label'):
        entity_type = chunk.label()
        counts[entity_type] = counts.get(entity_type, 0) + 1

# Print results
print("Entity counts:")
for entity, count in counts.items():
    print(f"{entity}: {count}")
print(f"Total entities: {sum(counts.values())}")