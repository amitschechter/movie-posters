import os
import pandas as pd

print('Starts loading data...')
project_path = os.path.abspath(os.path.join(__file__ ,"../../.."))
data_dir = os.path.join(project_path, 'Data')
file_path = os.path.join(data_dir, 'tmdb-5000-movie-dataset/tmdb_5000_movies.csv')
cols = ["id", "title", "genres"]
data = pd.read_csv(file_path, header=0, usecols=cols, index_col="id")
# print(data)
# To get a specific movie use: data.loc[id]

for row in data.iterrows():
	print row
	break 