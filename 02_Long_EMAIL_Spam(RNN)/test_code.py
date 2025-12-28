import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import pickle
import re
import string

# ----------------------------
# Load model from Hugging Face
# ----------------------------
def load_model():
    model_url = "https://huggingface.co/MrLeo0087/spam-email-classifier/resolve/main/email_rnn.keras"
    model_path = tf.keras.utils.get_file("email_rnn.keras", model_url)
    model = tf.keras.models.load_model(model_path)
    return model

# ----------------------------
# Load tokenizer from Hugging Face
# ----------------------------
def load_tokenizer():
    tokenizer_url = "https://huggingface.co/MrLeo0087/spam-email-classifier/resolve/main/tokenizer_email.pkl"
    r = requests.get(tokenizer_url)
    with open("tokenizer.pkl", "wb") as f:
        f.write(r.content)
    with open("tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    return tok

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
def predict_email(text, model, tokenizer, max_len=700):
    clean_text = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    pred = model.predict(padded)[0][0]
    label = "Spam" if pred >= 0.5 else "Not Spam"
    confidence = pred if pred >= 0.5 else 1 - pred
    return label, confidence

# ----------------------------
# Main test script
# ----------------------------
if __name__ == "__main__":
    model = load_model()
    tokenizer = load_tokenizer()
    
    # Example email texts for testing
    test_emails = [
        "Congratulations! You have won a $5000 reward. Click here to claim it.",
        "Please find attached the project report for review.",
        "Get rich quick! Limited time investment offer.",
        "Can we reschedule our meeting to next Monday?",
    ]
    
    for email in test_emails:
        label, confidence = predict_email(email, model, tokenizer)
        print(f"Email: {email}")
        print(f"Prediction: {label}, Confidence: {confidence*100:.2f}%")
        print("-" * 50)
