import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import time

# Parameters
VOCAB_SIZE = 10000  # Number of words to keep from the dataset
MAX_LEN = 500       # Maximum length of the sequences
EMBEDDING_DIM = 32  # Dimension of the word embeddings
LSTM_UNITS = 64     # Number of LSTM units
BATCH_SIZE = 64     # Batch size for training
EPOCHS = 5          # Number of training epochs

def load_and_preprocess_data():
    """Load and preprocess the IMDB dataset."""
    print("Loading IMDB dataset...")
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    
    # Convert word indices back to text for traditional ML models
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    def decode_review(text_indices):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in text_indices])
    
    # Convert sequences back to text for traditional ML
    X_train_text = [' '.join([str(i) for i in x]) for x in X_train]
    X_test_text = [' '.join([str(i) for i in x]) for x in X_test]
    
    # For LSTM
    X_train_pad = pad_sequences(X_train, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test, maxlen=MAX_LEN, padding='post', truncating='post')
    
    return (X_train_text, X_test_text, X_train_pad, X_test_pad, y_train, y_test)

def train_lstm(X_train, y_train, X_val, y_val):
    """Build and train the LSTM model."""
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    training_time = time.time() - start_time
    
    return model, history, training_time

def train_ml_model(model, X_train, y_train, X_test, model_name):
    """Train a traditional ML model and return results."""
    print(f"\nTraining {model_name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Predict and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"{model_name} - Test Accuracy: {accuracy:.4f} (Trained in {training_time:.2f} seconds)")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    return model, accuracy, training_time

def plot_results(results):
    """Plot the comparison of different models."""
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    times = [results[model]['time'] for model in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    ax1.bar(models, accuracies, color='skyblue')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    # Training time comparison
    ax2.bar(models, times, color='lightgreen')
    ax2.set_title('Training Time (seconds)')
    ax2.set_ylabel('Time (s)')
    for i, v in enumerate(times):
        ax2.text(i, v + 0.1, f"{v:.2f}s", ha='center')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("TensorFlow version:", tf.__version__)
    
    # Load and preprocess data
    X_train_text, X_test_text, X_train_pad, X_test_pad, y_train, y_test = load_and_preprocess_data()
    
    # Split training data into training and validation sets
    X_train_text, X_val_text, X_train_pad, X_val_pad, y_train, y_val = train_test_split(
        X_train_text, X_train_pad, y_train, test_size=0.2, random_state=42)
    
    # Vectorize text data for traditional ML models
    print("\nVectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to avoid memory issues
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_val_tfidf = vectorizer.transform(X_val_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    
    # Dictionary to store results
    results = {}
    
    # 1. Train and evaluate LSTM
    print("\n" + "="*50)
    print("Training LSTM Model")
    print("="*50)
    lstm_model, lstm_history, lstm_time = train_lstm(X_train_pad, y_train, X_val_pad, y_val)
    lstm_accuracy = lstm_model.evaluate(X_test_pad, y_test, verbose=0)[1]
    results['LSTM'] = {'accuracy': lstm_accuracy, 'time': lstm_time}
    
    # 2. Train and evaluate Naive Bayes
    print("\n" + "="*50)
    print("Training Naive Bayes Model")
    print("="*50)
    nb_model, nb_accuracy, nb_time = train_ml_model(
        MultinomialNB(), X_train_tfidf, y_train, X_test_tfidf, "Naive Bayes")
    results['Naive Bayes'] = {'accuracy': nb_accuracy, 'time': nb_time}
    
    # 3. Train and evaluate Logistic Regression
    print("\n" + "="*50)
    print("Training Logistic Regression Model")
    print("="*50)
    lr_model, lr_accuracy, lr_time = train_ml_model(
        LogisticRegression(max_iter=1000), X_train_tfidf, y_train, X_test_tfidf, "Logistic Regression")
    results['Logistic Regression'] = {'accuracy': lr_accuracy, 'time': lr_time}
    
    # 4. Train and evaluate SVM
    print("\n" + "="*50)
    print("Training SVM Model")
    print("="*50)
    svm_model, svm_accuracy, svm_time = train_ml_model(
        SVC(kernel='linear'), X_train_tfidf, y_train, X_test_tfidf, "SVM")
    results['SVM'] = {'accuracy': svm_accuracy, 'time': svm_time}
    
    # Print and plot comparison
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    for model, metrics in results.items():
        print(f"{model}: Accuracy = {metrics['accuracy']:.4f}, Training Time = {metrics['time']:.2f}s")
    
    plot_results(results)
    
    # Save the best model
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")
    
    if best_model_name == 'LSTM':
        lstm_model.save('best_sentiment_model.keras')
        print("Best model (LSTM) saved as 'best_sentiment_model.keras'")
    else:
        import joblib
        joblib.dump(eval(f"{best_model_name.lower().replace(' ', '_')}_model"), 'best_sentiment_model.joblib')
        print(f"Best model ({best_model_name}) saved as 'best_sentiment_model.joblib'")

if __name__ == "__main__":
    main()