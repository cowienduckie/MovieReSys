import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer

class ContentBasedRecommender:
    def __init__(self):
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
        tf = TfidfVectorizer(ngram_range=(1, 2), min_df=0, stop_words='english')
        tfidf_matrix = tf.fit_transform(self.smd['description'])
        self.cosine_sim = linear_kernel(tfidf_matrix)

    def prepareMetadataBased(self):
        try:
            credits = pd.read_csv('../input/movies-data/credits_small.csv')
        except FileNotFoundError:
            credits = pd.read_csv('../input/the-movies-dataset/credits.csv')
            links_small = pd.read_csv('../input/the-movies-dataset/links_small.csv')['tmdbId']
            links_small = links_small.dropna().astype(int)
            credits = credits[credits['id'].isin(links_small)]


            def get_director(x):
                for i in literal_eval(x):
                    if i['job'] == 'Director':
                        return i['name']
                return ''


            credits['crew'] = credits['crew'].apply(get_director)
            credits = credits.rename(columns={'crew': 'director'})
            credits['cast'] = credits['cast'].apply(lambda x: [i['name'] for i in literal_eval(x)[:3]])
            credits = credits.astype(str).drop_duplicates()

            keywords = pd.read_csv('../input/the-movies-dataset/keywords.csv')
            keywords = keywords[keywords['id'].isin(links_small)].drop_duplicates()
            keywords['keywords'] = keywords['keywords'].apply(lambda x: [i['name'] for i in literal_eval(x)])

            credits = keywords.astype(str).merge(credits)
            credits.to_csv('../input/movies-data/credits_small.csv', index=False)

        credits[['cast', 'keywords']] = credits[['cast', 'keywords']].applymap(literal_eval)
        credits.head()
        strip = lambda x: str(x).replace(" ", "").lower()
        cast = credits['cast'].apply(lambda x: [strip(i) for i in x])
        director = credits['director'].apply(lambda x: [strip(x)] * 3)
        s = pd.DataFrame(np.concatenate(credits['keywords'])).value_counts()
        s[:5]
        s = s[s > 1]
        stem = SnowballStemmer('english').stem
        def filter_keywords(x):
            words = []
            for i in x:
                if i in s:
                    for a in i.split():
                        words.append(stem(a))
            return words
        keywords = credits['keywords'].apply(filter_keywords)
        md = pd.read_csv('../input/movies-data/metadata_small.csv', dtype=
        {'id': int}, usecols=['id', 'genres'])
        genres = credits.merge(md)[['id', 'genres']].drop_duplicates()
        genres = genres['genres'].reset_index(drop=True).apply(literal_eval)
        soup = keywords + cast + director + genres
        soup = soup.apply(lambda x: ' '.join(x))
        soup.head()
        count = CountVectorizer(ngram_range=(1, 2), min_df=0, stop_words='english')
        count_matrix = count.fit_transform(soup)
        self.cosine_sim = cosine_similarity(count_matrix)

    def recommend(self, title):
        movie = self.smd[self.smd['title'] == title]
        if len(movie) > 1:
            print("There are duplications of same name. Choose index and use get_recommendations(idx)")
            print(movie)
        else:
            indexes = self.get_recommendations(movie.index[0])
            recommend_movies = self.smd.iloc[indexes]
            return recommend_movies[1:].set_index('id') 

    def get_recommendations(self, idx):    
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        return [i[0] for i in sim_scores if i[1] > 0.01]

    def improved_recommendations(self, title):
        movies = self.recommend(title)[:25]
        md_s = pd.read_csv('../input/movies-data/metadata_small.csv', dtype=
        {'id': int, 'vote_count': int, 'vote_averages': float})
        md_s = md_s[md_s['id'].isin(movies.index)]
        return self.weighted_rating(md_s, 0.6)

    def weighted_rating(self, df, percentile=0.95):
        C = df['vote_average'].mean()
        m = df['vote_count'].quantile(percentile)
        qualified = df[df['vote_count'] >= m].copy()
        R = qualified['vote_average']
        v = qualified['vote_count']
        qualified['weighted_rating'] = (v / (v + m) * R) + (m / (m + v) * C)
        qualified = qualified.sort_values('weighted_rating', ascending=False)
        return qualified

# recommender = ContentBasedRecommender()
# recommender.recommend('The Dark Knight').head(10)
# recommender.prepareMetadataBased()
# recommender.recommend('The Dark Knight').head(10)
# recommender.improved_recommendations('The Dark Knight').head(10)
