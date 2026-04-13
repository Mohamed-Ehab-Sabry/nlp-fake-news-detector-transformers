# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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


