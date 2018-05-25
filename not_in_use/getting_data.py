import urllib2
import requests
import json
import time
import itertools
# import wget
import os
import tmdbsimple as tmdb
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle



poster_folder='posters_final/'
if poster_folder.split('/')[0] in os.listdir('/Users/dhruvkedia/Desktop/'):
    print('Folder already exists')
else:
    os.mkdir('/Users/dhruvkedia/Desktop/'+poster_folder)
    print('Created Folder')

# For the purpose of this example, i will be working with the 1999 Sci-Fi movie - "The Matrix"!

api_key = '99259b4843bc81e609e39b4ac7905460' #Enter your own API key here to run the code below. 
# Generate your own API key as explained above :)


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

def get_movie_info_tmdb(movie):
    response = search.movie(query=movie)
    id=response['results'][0]['id']
    movie = tmdb.Movies(id)
    info=movie.info()
    return info

def get_movie_genres_tmdb(movie):
    response = search.movie(query=movie)
    id=response['results'][0]['id']
    movie = tmdb.Movies(id)
    genres=movie.info()['genres']
    return genres







all_movies=tmdb.Movies()
top1000_movies=[]
print('Pulling movie list, Please wait...')
for i in range(1,51):
    if i%15==0:
        time.sleep(7)
    movies_on_this_page=all_movies.popular(page=i)['results']
    top1000_movies.extend(movies_on_this_page)
len(top1000_movies)
print(top1000_movies)
f3=open('movie_list.pckl','wb')
pickle.dump(top1000_movies,f3)
f3.close()
print('Done!')



f3=open('movie_list.pckl','rb')
top1000_movies=pickle.load(f3)
f3.close()

def list2pairs(l):
    # itertools.combinations(l,2) makes all pairs of length 2 from list l.
    pairs = list(itertools.combinations(l, 2))
    # then the one item pairs, as duplicate pairs aren't accounted for by itertools
    for i in l:
        pairs.append([i,i])
    return pairs


allPairs = []
for movie in top1000_movies:
    allPairs.extend(list2pairs(movie['genre_ids']))
    
nr_ids = np.unique(allPairs)


movies = []
baseyear = 2017

print('Starting pulling movies from TMDB. If you want to debug, uncomment the print command. This will take a while, please wait...')
done_ids=[]
for g_id in nr_ids:
    #print('Pulling movies for genre ID '+g_id)
    baseyear -= 1
    for page in xrange(1,6,1):
        time.sleep(0.5)
    
        url = 'https://api.themoviedb.org/3/discover/movie?api_key=' + api_key
        url += '&language=en-US&sort_by=popularity.desc&year=' + str(baseyear) 
        url += '&with_genres=' + str(g_id) + '&page=' + str(page)

        data = urllib2.urlopen(url).read()

        dataDict = json.loads(data)
        movies.extend(dataDict["results"])
    done_ids.append(str(g_id))
print("Pulled movies for genres - "+','.join(done_ids))


movie_ids = [m['id'] for m in movies]
print "originally we had ",len(movie_ids)," movies"
movie_ids=np.unique(movie_ids)
print len(movie_ids)
seen_before=[]
no_duplicate_movies=[]
for i in range(len(movies)):
    movie=movies[i]
    id=movie['id']
    if id in seen_before:
        continue
#         print "Seen before"
    else:
        seen_before.append(id)
        no_duplicate_movies.append(movie)
print "After removing duplicates we have ",len(no_duplicate_movies), " movies"

f6=open("movies_for_posters",'wb')
pickle.dump(movies,f6)
f6.close()



poster_movies=[]
counter=0
movies_no_poster=[]
print("Total movies : ",len(movies))
print("Started downloading posters...")
for movie in movies:
    id=movie['id']
    title=movie['title']
    if counter==1:
        print('Downloaded first. Code is working fine. Please wait, this will take quite some time...')
    if counter%300==0 and counter!=0:
        print "Done with ",counter," movies!"
        print "Trying to get poster for ",title
    try:
        grab_poster_tmdb(title)
        poster_movies.append(movie)
    except:
        try:
            time.sleep(7)
            grab_poster_tmdb(title)
            poster_movies.append(movie)
        except:
            movies_no_poster.append(movie)
    counter+=1
print("Done with all the posters!")


f=open('poster_movies.pckl','r')
poster_movies=pickle.load(f)
f.close()