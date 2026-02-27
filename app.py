import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1. SET PAGE CONFIG FIRST (This fixes your error)
st.set_page_config(page_title="Sentiment Analysis", page_icon="🛒")

# 2. Then do other setup (NLTK downloads)
nltk.download('stopwords')
nltk.download('wordnet')

# 3. Define functions
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# 4. Load Model and Vectorizer
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_artifacts()

# 5. UI Layout
st.title("🛒 Flipkart Review Sentiment Analysis")

user_input = st.text_area("Enter Review Here:", "Type your review...")

if st.button("Predict"):
    if user_input:
        cleaned = clean_text(user_input)
        vec_input = vectorizer.transform([cleaned])
        prediction = model.predict(vec_input)[0]
        
        if prediction == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")
    else:
        st.warning("Please enter text.")