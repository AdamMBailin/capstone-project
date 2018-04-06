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


@app.route("/random")
def results():


            model = models.load_model('all_folk_model_network')
            word_to_index = pickle.load(open('folk_word_to_index.pkl', 'rb'))
            index_to_word = pickle.load(open('folk_index_to_word.pkl', 'rb'))
            seq_length = 20

    reverse_dictionary = index_to_word
    n_words = 500
    vocab_size = len(index_to_word)
    seed_text = np.random.choice(range(vocab_size), seq_length)
    input_text = [index_to_word[word] for word in seed_text]
    input_text = ' '.join(input_text)
    lyrics = functions.generate_song_from_text_seed(model, seq_length, reverse_dictionary, n_words, seed_text)
    lyrics = lyrics.replace('eol ', '<br/>').replace(' eol', '<br/>')
    lyrics = lyrics.replace(' eol ', '<br/>').replace(' eos ', '<br/><br/><br/>')
    lyrics = lyrics.split('<br/>')
    complete_song = lyrics
    print('New song finished!')
    return render_template('random_start_song.html', lyrics=complete_song)

@app.route("/")
def page():
   with open("templates/pages.html", 'r') as viz_file:
       return viz_file.read()

@app.route('/song_generator', methods=['POST', 'GET'])
def result():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':
        inputs = flask.request.form
        length = inputs['length']
        length = int(length)

        start = inputs['start']

            model = models.load_model('all_folk_model_network')
            word_to_index = pickle.load(open('folk_word_to_index.pkl', 'rb'))
            index_to_word = pickle.load(open('folk_index_to_word.pkl', 'rb'))
            seq_length = 20

        text_in = start.lower().replace(' n ', ' eol ').replace('[verse ', '[verse')
        text_in = text_in.replace("'", '').replace('-', ' ')
        tokens = text_in.split()
        table = str.maketrans('', '', string.punctuation)
        tokens = [word.translate(table) for word in tokens]
        seed_text = functions.generate_seed_text_from_input(tokens, word_to_index)

        lyrics = functions.generate_song_from_input(model, seq_length, index_to_word, length, seed_text)
        lyrics = lyrics.replace('eol ', '<br/>').replace(' eol', '<br/>')
        lyrics = lyrics.replace(' eol ', '<br/>').replace(' eos ', '<br/><br/><br/>')
        lyrics = lyrics.split('<br/>')
        complete_song = lyrics
        print('New song finished!')
        return render_template('your_start_song.html', lyrics=complete_song, start_text=start)

if __name__ == '__main__':
    app.run()
