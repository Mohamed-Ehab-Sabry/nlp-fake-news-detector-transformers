import pickle
from pathlib import Path
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = Path.cwd().parents[1]
DATA_DIR = BASE_DIR / "data"
FEATURES_DIR = DATA_DIR / "dl_features"
print(BASE_DIR)
print(DATA_DIR)
print(FEATURES_DIR)


# Functions

def load_features():
    """Load the preprocessed arrays and metadata."""
    print("Loading data from:", FEATURES_DIR)
    X_train_pad = np.load(FEATURES_DIR / "X_train_pad.npy")
    X_val_pad = np.load(FEATURES_DIR / "X_val_pad.npy")
    X_test_pad = np.load(FEATURES_DIR / "X_test_pad.npy")

    y_train = np.load(FEATURES_DIR / "y_train.npy")
    y_val = np.load(FEATURES_DIR / "y_val.npy")
    y_test = np.load(FEATURES_DIR / "y_test.npy")

    with open(FEATURES_DIR / "meta.pkl", "rb") as f:
        meta = pickle.load(f)

    return X_train_pad, X_val_pad, X_test_pad, y_train, y_val, y_test, meta

def build_cnn_model(max_words, max_len, embedding_dim=50, dropout_rate=0.5, optimizer='adam'):
    """
    Build the 1D-CNN model. 
    Notice: added dropout_rate and optimizer as arguments for tuning!
    """
    print(f"Building CNN model (Optimizer: {optimizer}, Dropout: {dropout_rate})....")
    model = Sequential([
        Input(shape=(max_len,)), 
        Embedding(input_dim=max_words, output_dim=embedding_dim),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics as a dictionary."""
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_labels = (y_pred_probs > 0.5).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred_labels),
        "precision": precision_score(y_test, y_pred_labels),
        "recall": recall_score(y_test, y_pred_labels),
        "f1": f1_score(y_test, y_pred_labels)
    }
    return metrics