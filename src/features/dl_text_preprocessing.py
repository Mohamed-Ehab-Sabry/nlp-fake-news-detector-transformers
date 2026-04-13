# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer

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
