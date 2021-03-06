def input_cleaner(text_in):
    text_in = text_in.lower().replace(' n ', ' eol ').replace('[verse ', '[verse')
    text_in = text_in.replace("'", '').replace('-', ' ')
    tokens = text_in.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [word.translate(table) for word in tokens]
    return tokens

def generate_seed_text_from_input():
    input_text = input('Type what you want here.')
    print('Rendering text as lyrics...')
    seed_text = input_cleaner(input_text)
    seed_text = [word_to_index[word] if word in word_to_index else word_to_index['unknown'] for word in seed_text]
    seed_text = pad_sequences([seed_text], maxlen=20, value=2500)
    return seed_text, input_text

def generate_song_from_input(model, seq_length, reverse_dictionary, n_words, seed_text):
    print('Writing new song')
    result = []
    seed_text = np.squeeze(seed_text, axis=0)
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

def generate_new_song(model):
    seq_length = 20
    reverse_dictionary = index_to_word
    n_words = int(input('How many words?'))
    seed_text, input_text = generate_seed_text_from_input()
    lyrics = generate_song_from_input(model, seq_length, reverse_dictionary, n_words, seed_text)
    lyrics = lyrics.replace(' eol ', ' \n ').replace(' eos ', ' \n\n\n ')
    complete_song = input_text + '\n' + lyrics
    print('New song finished!')
    return complete_song
