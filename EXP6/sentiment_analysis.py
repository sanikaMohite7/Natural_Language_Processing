import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# Install required packages (uncomment if needed)
# import sys
# import subprocess
# subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.24.3", "tensorflow==2.13.0", "matplotlib==3.7.1"])

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
    
    # Pad sequences to the same length
    X_train = pad_sequences(X_train, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test = pad_sequences(X_test, maxlen=MAX_LEN, padding='post', truncating='post')
    
    return (X_train, y_train), (X_test, y_test)

def build_model():
    """Build the sentiment analysis model."""
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
    
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model with early stopping and model checkpointing."""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        ModelCheckpoint('best_model.keras', save_best_only=True, save_weights_only=False)
    ]
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def plot_history(history):
    """Plot training and validation accuracy and loss."""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("TensorFlow version:", tf.__version__)
    
    # Load and preprocess data
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_data()
    
    # Split training data into training and validation sets
    val_size = int(0.2 * len(X_train))
    X_val, X_train = X_train[:val_size], X_train[val_size:]
    y_val, y_train = y_train[:val_size], y_train[val_size:]
    
    print(f"\nDataset sizes:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Build and train the model
    print("\nBuilding model...")
    model = build_model()
    model.summary()
    
    print("\nTraining model...")
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate the model
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plot_history(history)
    
    # Save the final model
    model.save('sentiment_analysis_model.keras')
    print("\nModel saved as 'sentiment_analysis_model.keras'")

if __name__ == "__main__":
    main()