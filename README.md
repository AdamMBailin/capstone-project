You will live forever you tomatoes - a natural language generation project
==========================================================================
## Introduction:
I have always had an interest in writing songs and poetry. I wanted to combine my passion for song writing with my new machine learning skills. I also wanted to learn more about neural networks, natural language generation, and flask application development. This is currently an ongoing project.

## Data:
I collected lists of pitchfork reviewed albums from [albumoftheyear](www.albumoftheyear.org) for several genres. For each album I gathered the track listing and lyrics for each song from [genius](https://genius.com/). The dataset contains 5942 folk songs. Approximately 205 of these folk songs did not have track counts and another 211 were instrumental songs. Instrumental songs and songs without trackcounts were dropped. Song lyrics were split into seqeunces of 20 to 25 words with each seqeunce shifted over one word from the previous sequence. I initially used keras to create LSTM RRNs, but I currently developing models with TensorFlow.

## Objectives:
The goals of this project are to:
1. To want to develop album, artist, and genre level models for folk music that generate new lyrics
2. To refine the models to output song lyrics with structure, consistent themes and rhymes
3. To expand model creation to more albums, artists and genres
4. To create a Flask application with basic design elements

## Progress:
1. Data Collection [notebook](https://github.com/AdamMBailin/capstone-project/blob/master/lyric_data_collection.ipynb)
2. Data combining [notebook](https://github.com/AdamMBailin/capstone-project/blob/master/lyrical_data_combining.ipynb)
3. Initial [EDA](https://github.com/AdamMBailin/capstone-project/blob/master/EDA.ipynb)
4. [Album level model](https://github.com/AdamMBailin/capstone-project/blob/master/in_the_aeroplane_over_the_sea_initial_models.ipynb) and text generation for *In the Aeroplane Over the Sea*
5. Iteration [1](https://github.com/AdamMBailin/capstone-project/blob/master/in_the_aeroplane_over_the_sea_other_initial_models.ipynb) and [2](https://github.com/AdamMBailin/capstone-project/blob/master/in_the_aeroplane_over_the_sea_other_padding_models.ipynb) of the album level model
6. [Artist level model](https://github.com/AdamMBailin/capstone-project/blob/master/the_decemberists_artist_model.ipynb) and text generation for The Decemberists
7. [Genre level model](https://github.com/AdamMBailin/capstone-project/blob/master/all_folk_genre_model.ipynb) for folk
8. [Folk model text generation](https://github.com/AdamMBailin/capstone-project/blob/master/all_folk_lyric_generation.ipynb)
