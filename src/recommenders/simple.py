import pandas as pd
from ast import literal_eval

# Simple Recommender show results base on votes of all Users
class SimpleRecommender:
    def __init__(self):
        try:
            self.md = pd.read_csv('../input/movies-data/metadata_small.csv', dtype=
            {'id': int, 'vote_count': int, 'vote_averages': float})
        except FileNotFoundError:
            cols = ['id', 'title', 'release_date', 'genres', 'vote_count',
                    'vote_average', 'popularity']
            self.md = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv',
                            skiprows=[19731, 29504, 35588],  
                            usecols=cols)
            
            self.md['genres'] = self.md['genres'].apply(lambda x: [i['name'] for i in literal_eval(x)])
            self.md = self.md[self.md['title'].notnull()].astype({'vote_count': int})
            self.md = self.md[cols]
            self.md.to_csv('../input/movies-data/metadata_small.csv', index=False)
    
    def get_category_list(self):
        gernes = self.md.genres.tolist()

        category_list = []

        for item in gernes:
            for g in item:
                category_list.append(g)

        return list(dict.fromkeys(category_list))

    
    def weighted_rating(self, df, percentile=0.95):
        C = df['vote_average'].mean()
        m = df['vote_count'].quantile(percentile)
        qualified = df[df['vote_count'] >= m].copy()
        R = qualified['vote_average']
        v = qualified['vote_count']
        qualified['weighted_rating'] = (v / (v + m) * R) + (m / (m + v) * C)
        qualified = qualified.sort_values('weighted_rating', ascending=False)
        return qualified

    def build_chart(self, genre, percentile=0.85):
        df = self.md[self.md.genres.apply(lambda x: genre in x)]
        return self.weighted_rating(df, percentile)


