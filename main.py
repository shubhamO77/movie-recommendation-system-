import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


data = pd.read_csv('movies.csv')  
usermovie = input("Enter the name of a movie: ")
moviefeatures = data.drop(['Movie Names', 'Genre'], axis=1)
scaler = StandardScaler()
content_features_scaled = scaler.fit_transform(moviefeatures)


knn_model = NearestNeighbors(n_neighbors=11, metric='euclidean')
knn_model.fit(content_features_scaled)
movie_index = data[data['Movie Names'] == usermovie].index[0]
distances, indices = knn_model.kneighbors(content_features_scaled[movie_index].reshape(1, -1), n_neighbors=11)
recommended_movie_indices = indices.flatten()[1:] 


print("Recommended movies:")
for idx in recommended_movie_indices:
    recommended_movie = data.iloc[idx]['Movie Names']
    recommended_rating = data.iloc[idx]['Rating']
    print(f"Movie name: {recommended_movie}, Rating: {recommended_rating}")
