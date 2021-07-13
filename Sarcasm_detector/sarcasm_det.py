# import tensorflow as tf
import numpy as np
import pandas as pd
import regex as re
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score


sar_acc = pd.read_json('Sarcasm_Headlines_Dataset.json',lines=True)
sar_acc['source'] = sar_acc['article_link'].apply(lambda x: re.findall(r'\w+', x)[2])


X = sar_acc.headline
Y = sar_acc.is_sarcastic
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
tk = tf.keras.preprocessing.text.Tokenizer(num_words=1000)
tk.fit_on_texts(X_train)
seqs = tk.texts_to_sequences(X_train)
max_len = 100
seqs_mat = tf.keras.preprocessing.sequence.pad_sequences(seqs,maxlen=max_len)

def model_def():
    inputs = tf.keras.layers.Input(name='inputs', shape=[max_len])
    layer = tf.keras.layers.Embedding(1000,50,input_length=max_len)(inputs)
    layer = tf.keras.layers.LSTM(64)(layer)
    layer = tf.keras.layers.Dense(256,name='FC1')(layer)
    layer = tf.keras.layers.Activation('relu')(layer)
    layer = tf.keras.layers.Dropout(0.2)(layer)
    layer = tf.keras.layers.Dense(1,name='out_layer')(layer)
    
    model = tf.keras.models.Model(inputs=inputs,outputs=layer)
    return model





