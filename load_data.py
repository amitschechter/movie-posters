try:
    import urllib.request as urllib2
except ImportError:
    import urllib2

import requests
import json
import time
import itertools
import os
import tmdbsimple as tmdb
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import csv

project_path = os.path.abspath(os.path.join(__file__ ,"../.."))
movie_posters_path = os.path.abspath(os.path.join(__file__ ,"../"))
poster_folder='posters_final_different_titles/'
api_key = '99259b4843bc81e609e39b4ac7905460' #Enter your own API key here to run the code below. 
# Generate your own API key as explained above :)

if poster_folder.split('/')[0] in os.listdir(movie_posters_path):
    print('Folder already exists')
else:
    os.mkdir(movie_posters_path+'/'+poster_folder)
    print('Created Folder')

tmdb.API_KEY = api_key #This sets the API key setting for the tmdb object
search = tmdb.Search() #this instantiates a tmdb "search" object which allows your to search for the movie

# These functions take in a string movie name i.e. like "The Matrix" or "Interstellar"
# What they return is pretty much clear in the name - Poster, ID , Info or genre of the Movie!
def grab_poster_tmdb(movie):
    response = search.movie(query=movie)
    id=response['results'][0]['id']
    movie = tmdb.Movies(id)
    posterp=movie.info()['poster_path']
    title=movie.info()['original_title']
    url='image.tmdb.org/t/p/original'+posterp
    title='_'.join(title.split(' '))
    strcmd='wget -O '+poster_folder+title+'.jpg '+url
    os.system(strcmd)

def get_movie_id_tmdb(movie):
    response = search.movie(query=movie)
    movie_id=response['results'][0]['id']
    return movie_id

print('Starts loading data...')
# project_path = os.path.abspath(os.path.join(__file__ ,"../.."))
data_dir = os.path.join(project_path, 'Data')
file_path = os.path.join(data_dir, 'tmdb-5000-movie-dataset/tmdb_5000_movies_different_titles.csv')
cols = ["id", "title", "genres"]
data = pd.read_csv(file_path, header=0, usecols=cols)

print('Finished reading CSV. Start fetching images')


poster_movies=[]
counter=0
movies_no_poster=[]
# print("Total movies : ",len(movies))
print("Started downloading posters...")
for index, row in data.iterrows():
    dmdb_id = row[1]
    title = row[2]
    if counter==1:
        print('Downloaded first. Code is working fine. Please wait, this will take quite some time...')
    if counter%300==0 and counter!=0:
        print("Done with ",counter," movies!")
        print("Trying to get poster for ",title)
    try:
        grab_poster_tmdb(title)
        poster_movies.append(title)
    except:
        try:
            time.sleep(7)
            grab_poster_tmdb(title)
            poster_movies.append(title)
        except:
            movies_no_poster.append(title)
    counter+=1

print("Done with all the posters!")
print(poster_movies)
print(movies_no_poster)

with open('posters', 'w') as fp:
    pickle.dump((poster_movies, movies_no_poster), fp)

