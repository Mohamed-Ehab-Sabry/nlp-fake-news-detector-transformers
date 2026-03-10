import re
import string
import nltk
from nltk.corpus import stopwords
from src.data.loader import load_data

# stopwords list 
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
if 'not' in stop_words:
    stop_words.remove('not')

def textCleaning(text):
    # lowercase
    text = text.lower()
    
    # remove urls
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # remove hashtags symbol only (#word -> word)
    text = re.sub(r'#', '', text)
    
    # remove retweet tag
    text = re.sub(r'\brt\b', '', text, flags=re.IGNORECASE)
    
    # remove html entities
    text = re.sub(r'&\w+;', '', text)
    
    # remove numbers
    text = re.sub(r'\d+', '', text)
    
    # remove punctuation
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    
    # remove emojis 
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # split text into words
    words = text.split()
    
    # remove stopwords
    cleaned_words = []
    for w in words:
        if w not in stop_words:
            cleaned_words.append(w)
    
    # join words 
    text = " ".join(cleaned_words)
    
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def clean_dataframe(df, column_name):
    # remove nulls
    df = df.dropna(subset=[column_name])
    
    # remove empty strings
    df = df[df[column_name].str.strip() != ""]
    
    # remove duplicates
    df = df.drop_duplicates(subset=[column_name])
    
    
    return df


def processed_data(filepath, output_file="processed.csv"):
  
  print("Loading data...")
  df = load_data(filepath)

  print("Cleaning text...")
  df['cleaned_text'] = df['text'].apply(textCleaning)
  df = df[['cleaned_text', 'target']]

  print("Removing nulls, empty values, and duplicates...")
  df = clean_dataframe(df, 'cleaned_text')

  print("Saving dataset...")
  df[['cleaned_text', 'target']].to_csv(output_file, index=False)

  print(f"Cleaned dataset saved as {output_file}")

  return df

