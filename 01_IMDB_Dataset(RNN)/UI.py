import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import pickle
import re
import string

st.set_page_config(page_title="IMDB Sentiment Analyzer", layout="centered")

st.title("🎬 IMDB Movie Review Sentiment Analyzer")

# ----------------------------
# Download and load Keras model from Hugging Face
# ----------------------------
@st.cache_resource
def load_model():
    model_url = "https://huggingface.co/MrLeo0087/imdb-movie-review-sentiment-analysis/resolve/main/IMDB.keras"
    model_path = tf.keras.utils.get_file("IMDB.keras", model_url)
    model = tf.keras.models.load_model(model_path)
    return model

# ----------------------------
# Download and load tokenizer from Hugging Face
# ----------------------------
@st.cache_resource
def load_tokenizer():
    tokenizer_url = "https://huggingface.co/MrLeo0087/imdb-movie-review-sentiment-analysis/resolve/main/tokenizer.pkl"
    r = requests.get(tokenizer_url)
    with open("tokenizer.pkl", "wb") as f:
        f.write(r.content)
    with open("tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    return tok

model = load_model()
tokenizer = load_tokenizer()

MAX_LEN = 300  # Same as training

# ----------------------------
# Preprocessing function
# ----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----------------------------
# Prediction function
# ----------------------------
def predict_sentiment(review):
    clean_review = preprocess_text(review)
    seq = tokenizer.texts_to_sequences([clean_review])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    pred = model.predict(padded)[0][0]
    label = "Positive" if pred >= 0.5 else "Negative"
    confidence = pred if pred >= 0.5 else 1 - pred
    return label, confidence

# ----------------------------
# Streamlit input and display
# ----------------------------
review = st.text_area("Enter your movie review here:", height=150)

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review to predict!")
    else:
        label, confidence = predict_sentiment(review)
        st.subheader("Prediction")
        st.markdown(f"**Sentiment:** {label}")
        st.markdown(f"**Confidence:** {confidence*100:.2f}%")
        st.markdown("### " + ("😄" if label=="Positive" else "😞"))
