# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path
import pickle
import numpy as np

# load cleaned dataset
def load_clean_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Split data 
def split_data(df):
    X = df["cleaned_text"]
    y = df["target"]
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size= 0.15/0.8,
        random_state= 42,
        stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# Tokenization
def build_tokenizer(X_train, max_words = 20000):
    tokenizer = Tokenizer(num_words = max_words, oov_token = "<OOV>")
    tokenizer.fit_on_texts(X_train)
    return tokenizer

# convert text to sequences
def text_to_sequence(tokenizer, X_train, X_val, X_test):
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    return X_train_seq, X_val_seq, X_test_seq

# Padding 
def apply_padding(X_train_seq, X_val_seq, X_test_seq, max_len = 100):
    X_train_pad = pad_sequences(X_train_seq, maxlen = max_len, padding = "post", truncating = "post")
    X_val_pad = pad_sequences(X_val_seq, maxlen = max_len, padding = "post", truncating = "post")
    X_test_pad = pad_sequences(X_test_seq, maxlen = max_len, padding = "post", truncating = "post")
    return X_train_pad, X_val_pad, X_test_pad

# project path
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "processed" / "processed.csv"
FEATURES_DIR = DATA_DIR / "dl_features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# save everything
def save_outputs(X_train_pad, X_val_pad, X_test_pad, y_train, y_val, y_test, tokenizer):

    # save tokenizer
    with open(FEATURES_DIR / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # save padded sequences
    np.save(FEATURES_DIR / "X_train_pad.npy", X_train_pad)
    np.save(FEATURES_DIR / "X_val_pad.npy", X_val_pad)
    np.save(FEATURES_DIR / "X_test_pad.npy", X_test_pad)

    # save labels
    np.save(FEATURES_DIR / "y_train.npy", y_train)
    np.save(FEATURES_DIR / "y_val.npy", y_val)
    np.save(FEATURES_DIR / "y_test.npy", y_test)

    # save meta info
    meta = {
        "max_len": 100,
        "max_words": 20000
    }
    with open(FEATURES_DIR / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    # save label mapping
    label_map = {0: "negative", 1: "positive"}
    with open(FEATURES_DIR / "label_map.pkl", "wb") as f:
        pickle.dump(label_map, f)

    print("All outputs saved successfully ")    

def main():
    df = load_clean_data(DATA_PATH)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    tokenizer = build_tokenizer(X_train)

    X_train_seq, X_val_seq, X_test_seq = text_to_sequence(tokenizer, X_train, X_val, X_test)

    X_train_pad, X_val_pad, X_test_pad = apply_padding(X_train_seq, X_val_seq, X_test_seq)

    save_outputs(X_train_pad, X_val_pad, X_test_pad, y_train, y_val, y_test, tokenizer)

    print("\nFeature engineering completed successfully!")
if __name__ == "__main__":
    main()
    