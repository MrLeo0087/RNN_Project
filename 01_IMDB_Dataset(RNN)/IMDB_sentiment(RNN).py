# %%
# import os
# import shutil

# # Create the folder in your Windows user directory
# kaggle_path = os.path.expanduser('~/.kaggle')
# if not os.path.exists(kaggle_path):
#     os.makedirs(kaggle_path)

# # Copy the file (make sure kaggle.json is in your current folder)
# shutil.copy('kaggle.json', os.path.join(kaggle_path, 'kaggle.json'))

# print("Kaggle folder set up successfully!")

# %%
# # The '!' tells VS Code to run this in your terminal/command prompt
# !kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# %%
import pandas as pd
import numpy as np


# %%
df = pd.read_csv('D:\\Learning\\Test\\RNN\\New\\IMDB Dataset.csv')

# %%
df.info()

# %%
df.isnull().sum()


# %%
df['sentiment'].value_counts()

# %%
y = df['sentiment']
X = df['review']

# %%
y = np.where(df['sentiment']=='positive',1,0)

# %% [markdown]
# #### Lowercase

# %%
X = X.str.lower()

# %% [markdown]
# #### Remove Punctuation

# %%
import string
X = X.str.replace('{string.punctuation}',' ',regex=True)  

# %% [markdown]
# #### Remove Number

# %%
X = X.str.replace(r'\d+',' ',regex=True)

# %% [markdown]
# #### Remove URL Links

# %%
X = X.str.replace(r'http\s+|www\s+|https\s+',' ',regex=True)

# %% [markdown]
# #### Remove Emoji and extra character

# %%
X = X.str.replace(r'^a-zA-Z0-9\s',' ',regex=True)

# %% [markdown]
# #### Remove BR tag

# %%
X = X.str.replace('<br />',' ',regex=True)


# %%
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# %%
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(X)

# %%
volcab_size = len(tokenizer.word_index)+1
X_seq = tokenizer.texts_to_sequences(X)

# %%
tokenizer.word_index

# %%
X[0]

# %%
seq_lengths = [len(x) for x in X_seq] 
count = 0 
seq_lengths_sorted = sorted(seq_lengths, reverse=True)
for i in seq_lengths:
    if i > 300:
        count+=1

count


# %%
max_len = 300

# %%
X_padded = pad_sequences(X_seq,maxlen=max_len,padding='post',truncating='post')

# %% [markdown]
# ## Model

# %%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

from tensorflow.keras.layers import Dropout
model = Sequential()
model.add(Embedding(input_dim=volcab_size, output_dim=100, input_shape=(max_len,)))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(150))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))


# %%
model.summary()

# %%
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %%
model.fit(X_padded, y, epochs=10, batch_size=64, validation_split=0.2)

# %%
import pickle

model.keras.save("IMDB.keras")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)


