import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Paragraph about climate change
paragraph = """
Climate change is one of the most pressing issues facing our planet today. 
Rising global temperatures are causing extreme weather events, melting ice caps, 
and threatening biodiversity. Scientists warn that immediate action is needed to 
reduce greenhouse gas emissions and transition to renewable energy sources. 
The consequences of inaction could be catastrophic for future generations.
"""

# Tokenize into sentences
sentences = sent_tokenize(paragraph)
print("Sentences:")

# Tokenize into words
words = word_tokenize(paragraph)
print("\nWords:", words)

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words and word.isalnum()]
print("\nFiltered Words (without stop words and punctuation):")
print(filtered_words)

# Apply stemming
ps = PorterStemmer()
stemmed_words = [ps.stem(word) for word in filtered_words]
print("\nStemmed Words:")
print(stemmed_words)