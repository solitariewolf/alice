import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import numpy as np

# Carregando os dados
dataset = tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = dataset.load_data(num_words=10000)

# Convertendo os dados para texto
word_index = dataset.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
train_data = [' '.join([reverse_word_index.get(i, '?') for i in sentence]) for sentence in train_data]
test_data = [' '.join([reverse_word_index.get(i, '?') for i in sentence]) for sentence in test_data]

# Tokenização
tokenizer = Tokenizer(num_words=5000, split=' ')
tokenizer.fit_on_texts(train_data)

# Salva o tokenizador
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Transforma o texto em sequências de números
X = tokenizer.texts_to_sequences(train_data)
X = pad_sequences(X)

# Cria o modelo
embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(5000, embed_dim, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

# Treina o modelo
Y = pd.get_dummies(train_labels).values
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.2, random_state = 42)
model.fit(X_train, Y_train, epochs = 5, batch_size=32, validation_data=(X_valid, Y_valid))

# Salva o modelo
model.save('model.h5')

