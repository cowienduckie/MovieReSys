import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


try:
    md = pd.read_csv('../input/movies-data/metadata_small.csv', dtype=
    {'id': int, 'vote_count': int, 'vote_averages': float})
except FileNotFoundError:
    cols = ['id', 'title', 'release_date', 'genres', 'vote_count',
            'vote_average', 'popularity']
    md = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv',
                     skiprows=[19731, 29504, 35588],  
                     usecols=cols)
    
    md['genres'] = md['genres'].apply(lambda x: [i['name'] for i in literal_eval(x)])
    md = md[md['title'].notnull()].astype({'vote_count': int})
    md = md[cols]
    md.to_csv('../input/movies-data/metadata_small.csv', index=False)

md.head()


def weighted_rating(df, percentile=0.95):
    C = df['vote_average'].mean()
    m = df['vote_count'].quantile(percentile)
    qualified = df[df['vote_count'] >= m].copy()
    R = qualified['vote_average']
    v = qualified['vote_count']
    qualified['weighted_rating'] = (v / (v + m) * R) + (m / (m + v) * C)
    qualified = qualified.sort_values('weighted_rating', ascending=False)
    return qualified

weighted_rating(md).head(10)


def build_chart(genre, percentile=0.85):
    df = md[md.genres.apply(lambda x: genre in x)]
    return weighted_rating(df, percentile)

build_chart('Romance').head(10)



try:  
    smd = pd.read_csv('../input/movies-data/description.csv')
except FileNotFoundError:
    md = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv',
                     skiprows=[19731, 29504, 35588],  
                     dtype={'id': int},
                     usecols=['title', 'id', 'overview', 'tagline'])
    links_small = pd.read_csv('../input/the-movies-dataset/links_small.csv')['tmdbId']
    links_small = links_small.dropna().astype(int)
    smd = md[md['id'].isin(links_small)].copy()
    smd['description'] = smd['overview'].fillna('') + ' ' + smd['tagline'].fillna('')
    smd = smd[['id', 'title', 'description']].drop_duplicates()
    smd.to_csv('../input/movies-data/description.csv', index=False)
    smd = smd.reset_index(drop=True)

smd.shape 


smd['description'] = smd['description'].fillna('')
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=0, stop_words='english')


tfidf_matrix = tf.fit_transform(smd['description'])
tfidf_matrix.shape


cosine_sim = linear_kernel(tfidf_matrix)
cosine_sim

def recommend(title):
    movie = smd[smd['title'] == title]
    if len(movie) > 1:
        print("There are duplications of same name. Choose index and use get_recommendations(idx)")
        print(movie)
    else:
        indexes = get_recommendations(movie.index[0])
        recommend_movies = smd.iloc[indexes]
        return recommend_movies[1:].set_index('id')


def get_recommendations(idx):
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return [i[0] for i in sim_scores if i[1] > 0.01]

recommend('The Godfather').head(10)
recommend('The Dark Knight').head(10)


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

from nltk.stem.snowball import SnowballStemmer

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

cosine_sim = cosine_similarity(count_matrix)





recommend('The Dark Knight').head(10)










recommend('Mean Girls').head(10)











def improved_recommendations(title):
    movies = recommend(title)[:25]
    md_s = pd.read_csv('../input/movies-data/metadata_small.csv', dtype=
    {'id': int, 'vote_count': int, 'vote_averages': float})
    md_s = md_s[md_s['id'].isin(movies.index)]
    return weighted_rating(md_s, 0.6)


improved_recommendations('The Dark Knight')





improved_recommendations('Mean Girls')














from surprise import Reader, Dataset, SVD
from surprise import accuracy
from surprise.model_selection import train_test_split

ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')
ratings.head()





data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], Reader())



trainset, testset = train_test_split(data, test_size=.25)


algo = SVD()


algo.fit(trainset)
predictions = algo.test(testset)


accuracy.rmse(predictions)





trainset = data.build_full_trainset()
algo.fit(trainset)







algo.predict(1, 302)
















id_map = pd.read_csv('../input/the-movies-dataset/links_small.csv',
                     usecols=['movieId', 'tmdbId'])
id_map = id_map.dropna().astype(int).set_index('tmdbId')


def hybrid(userid, title):
    movies = recommend(title)
    movies['est'] = [algo.predict(userid, id_map.loc[x]['movieId']).est for x in movies.index]
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)


hybrid(1, 'Avatar')


hybrid(500, 'Avatar')





















