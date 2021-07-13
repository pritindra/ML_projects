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

model = model_def()
model.summary()
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

es = tf.keras.callbacks.EarlyStopping(
    monitor='val_acc', 
    mode='max',
    patience=6
)

model.fit(seqs_mat,Y_train,batch_size=100,epochs=5,validation_split=0.1,callbacks=[es])

max_len = 150
def predict_sarcasm(user_seq):
#     prediction
    prob = model.predict(user_seq)
    probability = np.mean(prob, axis=0)

    if probability > 0.5:
        return("Sarcastic")
    elif probability < 0.5:
        return("Not Sarcastic")
    elif probability == 0.5:
        return("Neutral")

def user_text_processing(user_text):
    user_text = user_text.split()
    user_text = [word.lower() for word in user_text if word not in stopwords]
    user_text
    user_seq = np.array(user_text)
    user_seq = tk.texts_to_sequences(user_seq)
    user_seq = tf.keras.preprocessing.sequence.pad_sequences(user_seq,maxlen=max_len)

    return user_seq


test_sequences = tk.texts_to_sequences(X_test)
test_sequences_matrix = tf.keras.preprocessing.sequence.pad_sequences(test_sequences,maxlen=max_len)

accr = model.evaluate(test_sequences_matrix,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

user_text = 'state population to double by 2040, babies to blame'
user_seq = user_text_processing(user_text)
user_seq
prediction = predict_sarcasm(user_seq)
print(f"Sentence '{user_text}' is of '{prediction}' nature")




