import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def load_and_preprocess_data(filepath):
    """
    Loads data and preprocesses the review text.
    """
    try:
        # Load dataset
        df = pd.read_csv(filepath)
    except:
        # Creating a dummy dataframe if file not found for demonstration
        print("File not found. Using dummy data for demonstration.")
        data = {
            'Review text': ['Nice product, good quality', 'Bad quality, waste of money', 'Decent product', 'Terrible experience', 'Loved it! Durable and cheap'],
            'Ratings': [4, 1, 3, 1, 5]
        }
        df = pd.DataFrame(data)

    # 1. Drop rows with missing reviews
    df = df.dropna(subset=['Review text'])

    # 2. Create Sentiment Label (0: Negative, 1: Positive)
    # Filtering out neutral reviews (Rating 3)
    df = df[df['Ratings'] != 3]
    df['Sentiment'] = df['Ratings'].apply(lambda x: 1 if x > 3 else 0)

    # 3. Text Cleaning
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def clean_text(text):
        # Remove special characters and numbers
        text = re.sub('[^a-zA-Z]', ' ', str(text))
        # Convert to lowercase
        text = text.lower()
        # Tokenize and remove stopwords + lemmatize
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return " ".join(words)

    df['Cleaned_Review'] = df['Review text'].apply(clean_text)
    
    return df

# Example usage (This will be called in the training script)
if __name__ == "__main__":
    df = load_and_preprocess_data('data.csv')
    print(df[['Review text', 'Cleaned_Review', 'Sentiment']].head())