import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import pickle
import re
import string

st.set_page_config(page_title="Spam Email Detector", layout="centered")
st.title("📧 Spam Email Detector")

# ----------------------------
# Load Keras model from Hugging Face
# ----------------------------
@st.cache_resource
def load_model():
    model_url = "https://huggingface.co/MrLeo0087/spam-email-classifier/resolve/main/email_rnn.keras"
    model_path = tf.keras.utils.get_file("email_rnn.keras", model_url)
    model = tf.keras.models.load_model(model_path)
    return model

# ----------------------------
# Load tokenizer from Hugging Face
# ----------------------------
@st.cache_resource
def load_tokenizer():
    tokenizer_url = "https://huggingface.co/MrLeo0087/spam-email-classifier/resolve/main/tokenizer_email.pkl"
    r = requests.get(tokenizer_url)
    with open("tokenizer.pkl", "wb") as f:
        f.write(r.content)
    with open("tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    return tok

model = load_model()
tokenizer = load_tokenizer()
MAX_LEN = 700  # Same as training

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
def predict_email(text):
    clean_text = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    pred = model.predict(padded)[0][0]
    label = "Spam" if pred >= 0.5 else "Not Spam"
    confidence = pred if pred >= 0.5 else 1 - pred
    return label, confidence

# ----------------------------
# Streamlit input and display
# ----------------------------
email_text = st.text_area("Enter your email text here:", height=150)

if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("Please enter an email text to predict!")
    else:
        label, confidence = predict_email(email_text)
        st.subheader("Prediction")
        st.markdown(f"**Type:** {label}")
        st.markdown(f"**Confidence:** {confidence*100:.2f}%")
        emoji = "⚠️ Spam" if label == "Spam" else "✅ Not Spam"
        st.markdown(f"### {emoji}")
