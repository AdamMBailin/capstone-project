import flask
from flask import Flask, render_template
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.callbacks import ModelCheckpoint
import pickle
import pandas as pd
import collections
from nltk.corpus import words
import functions

app = flask.Flask(__name__)

@app.route("/", methods=['GET'])
def page():
    models = ['All Folk', 'The Decemberists', 'Neutral Milk Hotel']
    return render_template('pages.html', models=models)

@app.route("/random")
def results():

    model = models.load_model('models/all_folk/model.h5')
    word_to_index = pickle.load(open('models/all_folk/word_to_index.pkl', 'rb'))
    index_to_word = pickle.load(open('models/all_folk/index_to_word.pkl', 'rb'))
    seq_length = 20

    reverse_dictionary = index_to_word
    n_words = 200
    vocab_size = len(index_to_word)
    seed_text = np.random.choice(range(vocab_size), seq_length)
    input_text = [index_to_word[word] for word in seed_text]
    input_text = ' '.join(input_text)
    lyrics = functions.generate_song_from_text_seed(model, seq_length, reverse_dictionary, n_words, seed_text)
    lyrics = lyrics.replace('eol ', '<br/>').replace(' eol', '<br/>')
    lyrics = lyrics.replace(' eol ', '<br/>').replace(' eos ', '<br/><br/><br/>')
    lyrics = lyrics.split('<br/>')
    complete_song = lyrics
    return render_template('random_start_song.html', lyrics=complete_song)

@app.route('/song_generator', methods=['POST', 'GET'])
def result():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':
        inputs = flask.request.form
        model = inputs['model']
        length = inputs['length']
        length = int(length)
        start = inputs['start']

        if model == 'All Folk':
            model = models.load_model('models/all_folk/model.h5')
            word_to_index = pickle.load(open('models/all_folk/word_to_index.pkl', 'rb'))
            index_to_word = pickle.load(open('models/all_folk/index_to_word.pkl', 'rb'))
            seq_length = 20
            is_unknown = True
            size_of_vocab = len(word_to_index)

        if model == 'The Decemberists':
            model = models.load_model('models/the_decemberists/model.h5')
            word_to_index = pickle.load(open('models/the_decemberists/word_to_index.pkl', 'rb'))
            index_to_word = pickle.load(open('models/the_decemberists/index_to_word.pkl', 'rb'))
            seq_length = 25
            is_unknown = False
            size_of_vocab = len(word_to_index) - 1

        if model == 'Neutral Milk Hotel':
            model = models.load_model('models/neutral_milk_hotel/model.h5')
            word_to_index = pickle.load(open('models/neutral_milk_hotel/word_to_index.pkl', 'rb'))
            index_to_word = pickle.load(open('models/neutral_milk_hotel/index_to_word.pkl', 'rb'))
            seq_length = 25
            is_unknown = False
            size_of_vocab = len(word_to_index)

        text_in = start.lower().replace(' n ', ' eol ').replace('[verse ', '[verse')
        text_in = text_in.replace("'", '').replace('-', ' ')
        tokens = text_in.split()
        table = str.maketrans('', '', string.punctuation)
        tokens = [word.translate(table) for word in tokens]

        seed_text = functions.generate_seed_text_from_input(tokens, word_to_index, seq_length, size_of_vocab, is_unknown)
        lyrics = functions.generate_song_from_input(model, seq_length, index_to_word, length, size_of_vocab, seed_text)
        lyrics = lyrics.replace('eol ', '<br/>').replace(' eol', '<br/>')
        lyrics = lyrics.replace(' eol ', '<br/>').replace(' eos ', '<br/><br/><br/>')
        lyrics = lyrics.split('<br/>')
        complete_song = lyrics
        print('New song finished!')
        return render_template('your_start_song.html', lyrics=complete_song, start_text=start)

if __name__ == '__main__':
    app.run()
