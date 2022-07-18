import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise import accuracy
from surprise.model_selection import train_test_split

class CollaborativeBasedRecommender:
    def __init__(self):
        self.ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')
        data = Dataset.load_from_df(self.ratings[['userId', 'movieId', 'rating']], Reader())

        # sample random trainset and testset
        # test set is made of 25% of the ratings.
        trainset, testset = train_test_split(data, test_size=.25)

        # We'll use the famous SVD algorithm.
        self.algo = SVD()

        # Train the algorithm on the trainset, and predict ratings for the testset
        self.algo.fit(trainset)
        predictions = self.algo.test(testset)

        # Then compute RMSE
        accuracy.rmse(predictions)

        trainset = data.build_full_trainset()
        self.algo.fit(trainset)

        # Prepare dataset
        self.id_map = pd.read_csv('../input/the-movies-dataset/links_small.csv',
                     usecols=['movieId', 'tmdbId'])
        self.id_map = self.id_map.dropna().astype(int).set_index('tmdbId')

        try:  
            self.smd = pd.read_csv('../input/movies-data/description.csv')
        except FileNotFoundError:
            md = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv',
                            skiprows=[19731, 29504, 35588],  
                            dtype={'id': int},
                            usecols=['title', 'id', 'overview', 'tagline'])
            links_small = pd.read_csv('../input/the-movies-dataset/links_small.csv')['tmdbId']
            links_small = links_small.dropna().astype(int)
            self.smd = md[md['id'].isin(links_small)].copy()
            self.smd['description'] = self.smd['overview'].fillna('') + ' ' + self.smd['tagline'].fillna('')
            self.smd = self.smd[['id', 'title', 'description']].drop_duplicates()
            self.smd.to_csv('../input/movies-data/description.csv', index=False)
            self.smd = self.smd.reset_index(drop=True)
            self.smd['description'] = self.smd['description'].fillna('')
    
    def recommend(self, userid):
        movies = self.smd.set_index('id') 
        movies['est'] = [self.algo.predict(userid, self.id_map.loc[x]['movieId']).est for x in movies.index]
        movies = movies.sort_values('est', ascending=False)
        return movies

    def getUserRatings(self, userid):
        user_ratings = self.ratings[self.ratings['userId'] == userid]       
        
        movies = self.smd[self.smd['id'].isin(user_ratings['movieId'])]
        # movies.merge(user_ratings['rating'],  how = 'left',
        #         left_on = 'movieId', right_on = 'id')

        df = pd.merge(movies, user_ratings, how = 'left', left_on=['id'], right_on=['movieId']).dropna()
        df = df.set_index('id')
        df = df.drop(['userId', 'movieId', 'timestamp'], axis=1)        
        df = df.sort_values('rating', ascending=False)

        return df