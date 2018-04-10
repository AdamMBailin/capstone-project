def generate_song_from_text_seed(model, seq_length, reverse_dictionary, n_words, seed_text):
    print('Writing new song')
    result = []
    for _ in range(n_words):
        in_text = np.expand_dims(seed_text, axis=0)
        predict_prob = model.predict_proba(in_text)
        yhat = np.random.choice(range(10001), 1, p=predict_prob[0])
        if yhat == 10000:
            word = words.words()[np.random.randint(0,len(words.words()))]
        else:
            word = reverse_dictionary[yhat[0]]
        seed_text = np.append(seed_text, yhat)
        seed_text = seed_text[-seq_length:]
        result.append(word)
    return ' '.join(result)

def results():
    model         = models.load_model('models/all_folk/model.h5')
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
    lyrics = lyrics.replace('eol ', '\n').replace(' eol', '\n')
    lyrics = lyrics.replace(' eol ', '\n').replace(' eos ', '\n\n\n')
    complete_song = lyrics
    print('New song complete')
    return complete_song
