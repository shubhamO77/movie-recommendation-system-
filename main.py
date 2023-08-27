import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
import streamlit as st

df2 = pd.read_csv('tmdb_5000_movies.csv')

tfidf = TfidfVectorizer(stop_words='english')
df2['overview'] = df2['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df2['overview'])

cosine_sim_content = linear_kernel(tfidf_matrix, tfidf_matrix)

knn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
knn.fit(df2[['vote_average', 'vote_count']])

def get_recommendations(title, cosine_sim_content=cosine_sim_content, knn=knn):
    idx = df2[df2['title'] == title].index[0]

    # KNN recommendations
    knn_indices = knn.kneighbors(df2[['vote_average', 'vote_count']].iloc[idx].values.reshape(1, -1), return_distance=False)[0]
    knn_recommendations = df2.iloc[knn_indices]['title'].tolist()

    # content-based recommendations
    sim_scores = list(enumerate(cosine_sim_content[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    content_indices = [i[0] for i in sim_scores if i[0] != idx]  

    content_recommendations = df2.iloc[content_indices]['title'].tolist()

    combined_recommendations = content_recommendations[:7] + knn_recommendations[:7]

    return combined_recommendations

while True:
    usermovie = input("Enter a movie title: ")
    if usermovie in df2['title'].values:
        recommended_movies = get_recommendations(usermovie)
        print("Recommended movies based on", usermovie)
        for i, movie in enumerate(recommended_movies):
            print(f"{i + 1}. {movie}")
        break
    else:
        print("Invalid movie title. Please try again.")



st.title("Movie Recommender System")

usermovie = st.text_input("Enter a movie title:")

if usermovie:
    if usermovie in df2['title'].values:
        recommended_movies = get_recommendations(usermovie)
        st.write("Recommended movies based on", usermovie)
        for i, row in recommended_movies.iterrows():
            st.write(f"{i + 1}. {row['title']} ({row['release_date'][:4]})")
            st.image(f"https://image.tmdb.org/t/p/w300{row['poster_path']}", use_container_width=True)
    else:
        st.write("Invalid movie title. Please try again.")
        
