# # %%
# from google.colab import files
# files.upload()

# # %%
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json


# # %%
# !kaggle datasets download -d 'purusinghvi/email-spam-classification-dataset'

# # %%
# import zipfile
# with zipfile.ZipFile('/content/email-spam-classification-dataset.zip','r') as zip_ref:
#   zip_ref.extractall('/content/dataset')

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# %%
df = pd.read_csv('/content/dataset/combined_data.csv')

# %%
df

# %%
df['label'].value_counts()

# %%
df.shape

# %%
X = df['text']
y = df['label']

# %%
X = X.str.lower()

# %%
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer


# %%
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(X)

# %%
vocal_size = len(tokenizer.word_index)+1

# %%
X_seq = tokenizer.texts_to_sequences(X)

# %%
max_len = max(len(x) for x in X_seq)

# %%
max_len

# %%
len_seq = [len(x) for x in X_seq]
count = 0
for i in len_seq:
  if i > 700:
    count+=1

count

# %%
max_len = 700

# %%
from tensorflow.keras.preprocessing.sequence import pad_sequences
X_padded = pad_sequences(X_seq,maxlen=max_len,padding='post',truncating = 'post')

# %%
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding,GRU,Dense,Dropout,Input

# %%
inputs = Input(shape=(max_len,))
x = Embedding(input_dim=vocal_size,output_dim=200)(inputs)
x = GRU(150,return_sequences=True)(x)
x = Dropout(0.3)(x)

x = GRU(125)(x)
x = Dropout(0.2)(x)

outputs = Dense(1,activation='sigmoid')(x)

model = Model(inputs = inputs, outputs = outputs)


# %%
model.summary()

# %%
model.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'])

# %%
model.fit(X_padded,y,epochs=3,validation_split=0.2)

# %%
model.save('email_rnn.keras')

# %%
test =  ["""
 Thank you for registering for the upcoming Data Science and Machine Learning Workshop scheduled for next month. This email serves as confirmation of your successful registration and provides additional details to help you prepare.

The workshop will focus on practical applications of machine learning, including data preprocessing, model selection, and performance evaluation. Sessions will be conducted by experienced industry professionals and will include hands-on exercises designed to reinforce key concepts.

A detailed agenda, along with preparatory materials, will be shared one week prior to the event. We recommend reviewing the materials in advance to maximize your learning experience. Participants are encouraged to bring their own laptops with the required software installed.

If you have any dietary restrictions or accessibility requirements, please inform us by replying to this email no later than Friday. Our team is committed to ensuring a comfortable and inclusive environment for all attendees.

Should you have any questions regarding the schedule, location, or content, feel free to contact the organizing committee. We look forward to your active participation and hope you find the workshop both informative and engaging.
"""]

test_seq =tokenizer.texts_to_sequences(test)
test_padded = pad_sequences(test_seq,maxlen=max_len,padding='post',truncating = 'post')
y_pred = model.predict(test_padded)
if y_pred>=0.5:
  print(f"SPAM : {y_pred}")

else:
  print(f"NOT SPAM : {y_pred}")




# %%
import pickle
with open('tokenizer_email.pkl','wb') as f:
  pickle.dump(tokenizer,f)


