# Spam Email Classifier (GRU + LSTM)

This model predicts whether an email is **Spam** or **Non-Spam**.

---

## Model

- **Architecture:** Embedding + GRU (150)+GRU(125) + Dense
- **Trained on:** Labeled spam and non-spam emails dataset


### Try Model

* **Hugging Face Model:** [View Model](https://huggingface.co/MrLeo0087/spam-email-classifier/blob/main/email_rnn.keras)
* **Hugging Face tokenizer Model:** [View Model](https://huggingface.co/MrLeo0087/spam-email-classifier/blob/main/tokenizer_email.pkl)


## Dataset

This dataset is from kaggle [Link](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset)

---

## Input

- Raw text email (string)
- Convert to integer sequences using the same tokenizer
- Pad sequences to `max_len`

NOTE:

This model work more finely on long email (100-1000 word).

---

## Output

- Probability between 0–1
- Closer to 1 → Spam
- Closer to 0 → Non-Spam

## Social Media Link

Github : [Click](https://github.com/MrLeo0087)

Facebook : [Click](https://www.facebook.com/darshan.chaulagain.2025)

Instagram : [Click](https://www.instagram.com/iamdarshan_087/)

Linkedin : [Click](https://www.linkedin.com/in/darshan-chaulagain-399a893a0/)

---
