# Import libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump
from pathlib import Path
from nltk.corpus import wordnet

# NLTK downloads
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger') 

# load cleaned dataset
def load_clean_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Text preprocessing tools
stop_words = set(stopwords.words('english'))
stop_words.discard("not") 
lemmatizer = WordNetLemmatizer()

# Tokenization
def tokenize(text):
    return nltk.word_tokenize(text.lower())

# Stopwords Removal
def remove_stopwords(tokens):
    return [word for word in tokens if word.isalpha() and len(word) > 2 and word not in stop_words]


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN 


    
# Lemmatization
def lemmatize_tokens(tokens):
    pos_tags = nltk.pos_tag(tokens) 
    lem_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    return lem_tokens

# Text preprocessing pipeline
def preprocess_text(text):
    tokens = tokenize(text)
    tokens_no_stop = remove_stopwords(tokens)
    lem_tokens = lemmatize_tokens(tokens_no_stop)
    return " ".join(lem_tokens)

# Split data 
def split_data(df):
    X = df["cleaned_text"]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

# preprocess_split
def preprocess_split(X_train, X_test):

    X_train_processed = X_train.apply(preprocess_text)
    X_test_processed = X_test.apply(preprocess_text)

    return X_train_processed, X_test_processed

# Initialize and configure the TF-IDF Vectorizer
def build_tfidf_features(X_train, X_test):
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1,2),
        min_df= 2,
        max_df=0.9,
        sublinear_tf=True,
    )

    # Fit the vectorizer on the training data and transform both training and testing sets
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, vectorizer

# project path
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "processed" / "processed.csv"
FEATURES_DIR = DATA_DIR / "saved_features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# save features
def save_features(X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer):

    dump(vectorizer, FEATURES_DIR / "tfidf_vectorizer.joblib")
    dump(X_train_tfidf, FEATURES_DIR / "X_train_tfidf.joblib")
    dump(X_test_tfidf, FEATURES_DIR / "X_test_tfidf.joblib")
    dump(y_train, FEATURES_DIR / "y_train.joblib")
    dump(y_test, FEATURES_DIR / "y_test.joblib")

    print("TF-IDF vectorizer and feature matrices saved successfully")
    print("X_train shape:", X_train_tfidf.shape)
    print("X_test shape:", X_test_tfidf.shape)

# main
def main():

    df = load_clean_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test = preprocess_split(X_train, X_test)
    X_train_tfidf, X_test_tfidf, vectorizer = build_tfidf_features(X_train, X_test)

    save_features(X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer)
    print("\nFeature engineering completed successfully!")

if __name__ == "__main__":
    main()