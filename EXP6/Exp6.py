import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


data = {
    "text": [
        "I love this movie, it was fantastic!",
        "This film was terrible and boring.",
        "Amazing storyline and brilliant acting.",
        "Worst experience ever, very disappointing.",
        "The movie was okay, not too good, not too bad."
    ],
    "label": ["positive", "negative", "positive", "negative", "neutral"]
}

df = pd.DataFrame(data)


def preprocess(text):
    text = text.lower()                                   
    text = "".join([c for c in text if c not in string.punctuation])  
    words = [w for w in text.split() if w not in stop_words]          
    return " ".join(words)

df["clean_text"] = df["text"].apply(preprocess)

print("Preprocessed Data:\n", df)


X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.3, random_state=42
)


count_vect = CountVectorizer()
X_train_bow = count_vect.fit_transform(X_train)
X_test_bow = count_vect.transform(X_test)

model_bow = MultinomialNB()
model_bow.fit(X_train_bow, y_train)
y_pred_bow = model_bow.predict(X_test_bow)

print("\nBag-of-Words Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_bow))
print(classification_report(y_test, y_pred_bow))

tfidf_vect = TfidfVectorizer()
X_train_tfidf = tfidf_vect.fit_transform(X_train)
X_test_tfidf = tfidf_vect.transform(X_test)

model_tfidf = MultinomialNB()
model_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)

print("\nTF-IDF Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_tfidf))
print(classification_report(y_test, y_pred_tfidf))
