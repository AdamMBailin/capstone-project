import string

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import models
from keras import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.callbacks import ModelCheckpoint

import pickle
import numpy as np
import pandas as pd
import collections


def load_data(filename='all_songs.csv', col='album', col_value='In the Aeroplane Over the Sea'):
    df    = pd.read_csv('all_songs.csv')
    df    = df.loc[df[col] == col_value]
    songs = df['lyrics'].values
    return songs

def lyric_cleaner(songs):
    lyric_tokens = []
    for song in songs:
        text = song.lower().replace(' n ', ' eol ').replace('[verse ', '[verse')
        text = text.replace("'", '').replace('-', ' ')
        tokens = text.split()
        table = str.maketrans('', '', string.punctuation)
        tokens = [word.translate(table) for word in tokens]
        lyric_tokens.append(tokens)
    return lyric_tokens

def lyric_gatherer(lyric_tokens):
    lyrics = []
    for song in lyric_tokens:
        song.append('eos')
        for lyric in song:
            lyrics.append(lyric)
    return lyrics

def vocabulary_dictionary(lyrics, n_vocab):
    word_count = collections.Counter(lyrics)
    most_common = word_count.most_common(n=n_vocab)
    vocab = []
    for word, count in most_common:
        vocab.append(word)
    word_to_index = dict(zip(vocab, range(0, len(vocab))))
    word_to_index['unknown'] = len(vocab)
    index_to_word = dict([(index, word) for word, index in word_to_index.items()])
    return word_to_index, index_to_word

def tokenizer(dictionary, lyrics):
    encoded_lyrics = [dictionary[lyric] if lyric in dictionary else dictionary['unknown'] for lyric in lyrics]
    return encoded_lyrics

def sequenizer(encoded_lyrics, seq_length):
    length = seq_length + 1
    sequences = []
    for i in range(length, len(encoded_lyrics)):
        sequence = encoded_lyrics[i-length:i]
        sequences.append(sequence)
    n_patterns = len(sequences)
    sequences = np.array(sequences)
    return sequences, n_patterns

def prepare_data(sequences):
    X, y = sequences[:, :-1], sequences[:, -1]
    return X, y

def prepare_model(vocab_size, seq_length, lstm_hidden_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(lstm_hidden_size, return_sequences=True))
    model.add(LSTM(lstm_hidden_size, return_sequences=False))
    model.add(Dense(lstm_hidden_size, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

seq_length = 20
n_vocab = 10000
vocab_size = n_vocab + 1
lstm_hidden_size = 50

data         = load_data(filename='all_songs.csv', col='is_folk', col_value=1)

lyric_tokens = lyric_cleaner(data)

lyrics       = lyric_gatherer(lyric_tokens)

word_to_index, index_to_word = vocabulary_dictionary(lyrics, n_vocab)

encoded_lyrics = tokenizer(word_to_index, lyrics)

sequences, n_patterns = sequenizer(encoded_lyrics, seq_length)

X, y = prepare_data(sequences)

model = prepare_model(vocab_size, seq_length, lstm_hidden_size)

X, X_test, y, y_test = train_test_split(X, y, train_size=0.5, random_state=2018)

y = to_categorical(y)

history = model.fit(X, y, batch_size=128, epochs=100, callbacks=callbacks_list)
