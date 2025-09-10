import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt_tab')
nltk.download('stopwords')


text = "Natural Language Processing is an exciting field. It includes tokenization, stemming, and removing stop words."

sentences = sent_tokenize(text)
words = word_tokenize(text)
print("Sentences:\n", sentences)
print("Words:\n", words)
 
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]
print("Filtered Words:\n", filtered_words)

ps = PorterStemmer()
stemmed_words = [ps.stem(word) for word in filtered_words]
print ("Stemmed Words:\n", stemmed_words)
