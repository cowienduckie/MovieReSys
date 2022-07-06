import pandas as pd
from contentBased import ContentBasedRecommender
from collaborativeBased import CollaborativeBasedRecommender

class HybridRecommender:
    def __init__(self):
        self.id_map = pd.read_csv('../input/the-movies-dataset/links_small.csv',
                     usecols=['movieId', 'tmdbId'])
        self.id_map = self.id_map.dropna().astype(int).set_index('tmdbId')
        self.contentBased = ContentBasedRecommender()
        self.collabBased = CollaborativeBasedRecommender()

    def hybrid(self, userid, title):
        movies = self.contentBased.recommend(title)
        movies['est'] = [self.collabBased.algo.predict(userid, self.id_map.loc[x]['movieId']).est for x in movies.index]
        movies = movies.sort_values('est', ascending=False)
        return movies.head(10)

recommender = HybridRecommender()
recommender.hybrid(1, 'Avatar')