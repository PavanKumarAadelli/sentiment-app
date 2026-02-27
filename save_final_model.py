import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Load Data
print("Loading data...")
df = pd.read_csv('data.csv')
df = df.dropna(subset=['Review text'])
df = df[df['Ratings'] != 3]
df['Sentiment'] = df['Ratings'].apply(lambda x: 1 if x > 3 else 0)
df['Cleaned_Review'] = df['Review text'].apply(clean_text)

# Vectorizer
print("Training Vectorizer...")
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['Cleaned_Review'])
y = df['Sentiment']

# Model
print("Training Model...")
model = LogisticRegression()
model.fit(X, y)

# Save files
print("Saving model.pkl and vectorizer.pkl...")
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Done! Files saved.")