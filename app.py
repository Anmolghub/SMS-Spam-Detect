import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [ps.stem(i) for i in text if i.isalnum() and i not in stopwords.words('english')]
    return " ".join(y)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter your message")

if st.button("Predict"):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    proba = model.predict_proba(vector_input)[0]

    if result == 1:
        st.error("🚫 Spam")
    else:
        st.success("✅ Not Spam")

    st.write(f"Confidence: {round(max(proba)*100, 2)}%")
