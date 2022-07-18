import pandas as pd
from recommenders.contentBased import *
from recommenders.collaborativeBased import *

class HybridRecommender:
    def __init__(self):
        self.id_map = pd.read_csv('../input/the-movies-dataset/links_small.csv',
                     usecols=['movieId', 'tmdbId'])
        self.id_map = self.id_map.dropna().astype(int).set_index('tmdbId')
        self.contentBased = ContentBasedRecommender()
        self.collabBased = CollaborativeBasedRecommender()

    def recommend(self, userid, title):
        movies = self.contentBased.recommendByMetadata(title)
        movies['est'] = [self.collabBased.algo.predict(userid, self.id_map.loc[x]['movieId']).est for x in movies.index]
        movies = movies.sort_values('est', ascending=False)
        return movies
