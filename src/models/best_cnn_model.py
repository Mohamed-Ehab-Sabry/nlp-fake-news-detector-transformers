import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_DIR = BASE_DIR / "data" / "dl_features"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True) 


def load_features():
    print(f"Loading features from: {FEATURES_DIR}")
    X_train = np.load(FEATURES_DIR / "X_train_pad.npy")
    X_val = np.load(FEATURES_DIR / "X_val_pad.npy")
    X_test = np.load(FEATURES_DIR / "X_test_pad.npy")
    y_train = np.load(FEATURES_DIR / "y_train.npy")
    y_val = np.load(FEATURES_DIR / "y_val.npy")
    y_test = np.load(FEATURES_DIR / "y_test.npy")
    with open(FEATURES_DIR / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return X_train, X_val, X_test, y_train, y_val, y_test, meta

def build_cnn_model(max_words, max_len, embedding_dim=50, dropout_rate=0.5, optimizer='adam'):
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
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, meta = load_features()

    BEST_CONFIG = {
        "name": "Large Batch(Adam)",
        "opt": "adam",
        "dr": 0.5,
        "bs": 64,
        "epochs": 10
    }

    print(f"\nTraining the Best Model: {BEST_CONFIG['name']}...")

    model = build_cnn_model(
        max_words=meta["max_words"],
        max_len=meta["max_len"],
        dropout_rate=BEST_CONFIG["dr"],
        optimizer=BEST_CONFIG["opt"]
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=BEST_CONFIG["epochs"],
        batch_size=BEST_CONFIG["bs"],
        callbacks=[early_stopping],
        verbose=1
    )

    print("\nFinal Evaluation on Test Set:")
    metrics = evaluate_model(model, X_test, y_test)
    for key, value in metrics.items():
        print(f"- {key.capitalize()}: {value:.4f}")

    