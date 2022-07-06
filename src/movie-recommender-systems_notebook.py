# %% [markdown]
# # Movies Recommender System
# 
# In this notebook, I will attempt at implementing a few recommendation algorithms (content based, popularity based and collaborative filtering) and try to build an ensemble of these models to come up with our final recommendation system. With us, we have two MovieLens datasets.
# 
# * **The Full Dataset:** Consists of 26,000,000 ratings and 750,000 tag applications applied to 45,000 movies by 270,000 users. Includes tag genome data with 12 million relevance scores across 1,100 tags.
# * **The Small Dataset:** Comprises 100,000 ratings and 1,300 tag applications applied to 9,000 movies by 700 users.
# 
# I will build a Simple Recommender using movies from the *Full Dataset* whereas all personalised recommender systems will make use of the small dataset (due to the computing power I possess being very limited). As a first step, I will build my simple recommender system.

# %%
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# %% [markdown]
# ## Simple Recommender
# 
# The Simple Recommender offers generalized recommendations to every user based on movie popularity and (sometimes) genre. The basic idea behind this recommender is that movies that are more popular and more critically acclaimed will have a higher probability of being liked by the average audience. This model does not give personalized recommendations based on the user.
# 
# The implementation of this model is extremely trivial. All we have to do is sort our movies based on ratings and popularity and display the top movies of our list. As an added step, we can pass in a genre argument to get the top movies of a particular genre. 

# %%
try:
    md = pd.read_csv('../input/movies-data/metadata_small.csv', dtype=
    {'id': int, 'vote_count': int, 'vote_averages': float})
except FileNotFoundError:
    cols = ['id', 'title', 'release_date', 'genres', 'vote_count',
            'vote_average', 'popularity']
    md = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv',
                     skiprows=[19731, 29504, 35588],  #skip error data
                     usecols=cols)
    #extract genres
    md['genres'] = md['genres'].apply(lambda x: [i['name'] for i in literal_eval(x)])
    md = md[md['title'].notnull()].astype({'vote_count': int})
    md = md[cols]
    md.to_csv('../input/movies-data/metadata_small.csv', index=False)

md.head()

# %% [markdown]
# I use the TMDB Ratings to come up with our **Top Movies Chart.** I will use IMDB's *weighted rating* formula to construct my chart. Mathematically, it is represented as follows:
# 
# Weighted Rating (WR) = $(\frac{v}{v + m} . R) + (\frac{m}{v + m} . C)$
# 
# where,
# * *v* is the number of votes for the movie
# * *m* is the minimum votes required to be listed in the chart
# * *R* is the average rating of the movie
# * *C* is the mean vote across the whole report
# 
# The next step is to determine an appropriate value for *m*, the minimum votes required to be listed in the chart. We will use **95th percentile** as our cutoff. In other words, for a movie to feature in the charts, it must have more votes than at least 95% of the movies in the list.
# I will build our overall Top 250 Chart and will define a function to build charts for a particular genre. Let's begin!

# %%
print(f"C = {md['vote_average'].mean()}")
print(f"m95 = {md['vote_count'].quantile(0.95)}")
md[md['vote_count'] >= 434].copy().shape

# %% [markdown]
# Therefore, to qualify to be considered for the chart, a movie has to have at least **434 votes** on TMDB. We also see that the average rating for a movie on TMDB is **5.618** on a scale of 10. And **2274** Movies qualify to be on our chart.

# %%
def weighted_rating(df, percentile=0.95):
    C = df['vote_average'].mean()
    m = df['vote_count'].quantile(percentile)
    qualified = df[df['vote_count'] >= m].copy()
    R = qualified['vote_average']
    v = qualified['vote_count']
    qualified['weighted_rating'] = (v / (v + m) * R) + (m / (m + v) * C)
    qualified = qualified.sort_values('weighted_rating', ascending=False)
    return qualified

# %% [markdown]
# ### Top Movies

# %%
weighted_rating(md).head(10)

# %% [markdown]
# We see that three Crime Drama, **The Shawshank Redemption**, **The Godfather** and **The Dark Knight** occur at the very top of our chart. The chart also indicates a strong bias of TMDB Users towards particular genres and directors.
# 
# Let us now construct our function that builds charts for particular genres. For this, we will use relax our default conditions to the **85** percentile instead of 95.

# %%
def build_chart(genre, percentile=0.85):
    df = md[md.genres.apply(lambda x: genre in x)]
    return weighted_rating(df, percentile)

# %% [markdown]
# Let us see our method in action by displaying the Top 10 Romance Movies (Romance almost didn't feature at all in our Generic Top Chart despite  being one of the most popular movie genres).
# ### Top Romance Movies

# %%
build_chart('Romance').head(10)

# %% [markdown]
# The top romance movie according to our metrics is Bollywood's **Dilwale Dulhania Le Jayenge**. This Shahrukh Khan starrer also happens to be one of my personal favorites.

# %% [markdown]
# ## Content Based Recommender
# 
# The recommender we built in the previous section suffers some severe limitations. For one, it gives the same recommendation to everyone, regardless of the user's personal taste. If a person who loves romantic movies (and hates action) were to look at our Top 15 Chart, s/he wouldn't probably like most of the movies. If s/he were to go one step further and look at our charts by genre, s/he wouldn't still be getting the best recommendations.
# 
# For instance, consider a person who loves *Dilwale Dulhania Le Jayenge*, *My Name is Khan* and *Kabhi Khushi Kabhi Gham*. One inference we can obtain is that the person loves the actor Shahrukh Khan and the director Karan Johar. Even if s/he were to access the romance chart, s/he wouldn't find these as the top recommendations.
# 
# To personalise our recommendations more, I am going to build an engine that computes similarity between movies based on certain metrics and suggests movies that are most similar to a particular movie that a user liked. Since we will be using movie metadata (or content) to build this engine, this also known as **Content Based Filtering.**
# 
# I will build two Content Based Recommenders based on:
# * Movie Overviews and Taglines
# * Movie Cast, Crew, Keywords and Genre
# 
# Also, as mentioned in the introduction, I will be using a subset of all the movies available to us due to limiting computing power available to me. 

# %%
try:  #small_movies_data
    smd = pd.read_csv('../input/movies-data/description.csv')
except FileNotFoundError:
    md = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv',
                     skiprows=[19731, 29504, 35588],  #skip error data
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

# %% [markdown]
# 
# We have **9082** movies available in our small movies' metadata dataset which is 5 times smaller than our original dataset of 45000 movies.

# %% [markdown]
# ### Movie Description Based Recommender
# 
# Let us first try to build a recommender using movie descriptions and taglines. We do not have a quantitative metric to judge our machine's performance so this will have to be done qualitatively.

# %%
smd['description'] = smd['description'].fillna('')
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=0, stop_words='english')

# %% [markdown]
# #### TF-IDF
# [TF-IDF wiki](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
# 
# The weighting scheme of $tf$ is nature raw count
# 
# $idf = \ln {\frac{1+n}{1+df(t)}}+1 $
# 
# Then normalize to unit vector

# %%
tfidf_matrix = tf.fit_transform(smd['description'])
tfidf_matrix.shape

# %% [markdown]
# #### Cosine Similarity
# 
# I will be using the Cosine Similarity to calculate a numeric quantity that denotes the similarity between two movies. Mathematically, it is defined as follows:
# 
# $cosine(x,y) = \frac{x. y^\intercal}{||x||.||y||} $
# 
# Since we have used the TF-IDF Vectorizer, calculating the Dot Product will directly give us the Cosine Similarity Score. Therefore, we will use sklearn's **linear_kernel** instead of cosine_similarities since it is much faster.

# %%
cosine_sim = linear_kernel(tfidf_matrix)
cosine_sim

# %% [markdown]
# We now have a pairwise cosine similarity matrix for all the movies in our dataset. The next step is to write a function that returns the 30 most similar movies based on the cosine similarity score.
# 

# %%
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
    # return movies index which similarity score bigger than 0.01
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return [i[0] for i in sim_scores if i[1] > 0.01]

# %% [markdown]
# We're all set. Let us now try and get the top recommendations for a few movies and see how good the recommendations are.

# %%
recommend('The Godfather').head(10)

# %%
recommend('The Dark Knight').head(10)

# %% [markdown]
# We see that for **The Dark Knight**, our system is able to identify it as a Batman film and subsequently recommend other Batman films as its top recommendations. But unfortunately, that is all this system can do at the moment. This is not of much use to most people as it doesn't take into considerations very important features such as cast, crew, director and genre, which determine the rating and the popularity of a movie. Someone who liked **The Dark Knight** probably likes it more because of Nolan and would hate **Batman Forever** and every other substandard movie in the Batman Franchise.
# 
# Therefore, we are going to use much more suggestive metadata than **Overview** and **Tagline**. In the next subsection, we will build a more sophisticated recommender that takes **genre**, **keywords**, **cast** and **crew** into consideration.

# %% [markdown]
# ### Metadata Based Recommender
# 
# To build our standard metadata based content recommender, we will need to merge our current dataset with the crew and the keyword datasets. Let us prepare this data as our first step.

# %%
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

# %% [markdown]
# We now have our cast, crew, genres and credits, all in one dataframe. Let us wrangle this a little more using the following intuitions:
# 
# 1. **Crew:** From the crew, we will only pick the director as our feature since the others don't contribute that much to the *feel* of the movie.
# 2. **Cast:** Choosing Cast is a little more tricky. Lesser known actors and minor roles do not really affect people's opinion of a movie. Therefore, we must only select the major characters and their respective actors. Arbitrarily we will choose the top 3 actors that appear in the credits list. 
# 

# %%
credits[['cast', 'keywords']] = credits[['cast', 'keywords']].applymap(literal_eval)
credits.head()

# %% [markdown]
# My approach to building the recommender is going to be extremely *hacky*. What I plan on doing is creating a metadata dump for every movie which consists of **genres, director, main actors and keywords.** I then use a **Count Vectorizer** to create our count matrix as we did in the Description Recommender. The remaining steps are similar to what we did earlier: we calculate the cosine similarities and return movies that are most similar.
# 
# These are steps I follow in the preparation of my genres and credits data:
# 1. **Strip Spaces and Convert to Lowercase** from all our features. This way, our engine will not confuse between **Johnny Depp** and **Johnny Galecki.** 
# 2. **Mention Director 3 times** to give it more weight relative to the entire cast.

# %%
strip = lambda x: str(x).replace(" ", "").lower()
cast = credits['cast'].apply(lambda x: [strip(i) for i in x])
director = credits['director'].apply(lambda x: [strip(x)] * 3)

# %% [markdown]
# #### Keywords
# 
# We will do a small amount of pre-processing of our keywords before putting them to any use. As a first step, we calculate the frequenct counts of every keyword that appears in the dataset.

# %%
s = pd.DataFrame(np.concatenate(credits['keywords'])).value_counts()
s[:5]

# %% [markdown]
# Keywords occur in frequencies ranging from 1 to 603. We do not have any use for keywords that occur only once. Therefore, these can be safely removed. Finally, we will convert every word to its stem so that words such as *Dogs* and *Dog* are considered the same.

# %%
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

# %%
keywords = credits['keywords'].apply(filter_keywords)
md = pd.read_csv('../input/movies-data/metadata_small.csv', dtype=
{'id': int}, usecols=['id', 'genres'])
genres = credits.merge(md)[['id', 'genres']].drop_duplicates()
genres = genres['genres'].reset_index(drop=True).apply(literal_eval)
soup = keywords + cast + director + genres
soup = soup.apply(lambda x: ' '.join(x))
soup.head()

# %%
count = CountVectorizer(ngram_range=(1, 2), min_df=0, stop_words='english')
count_matrix = count.fit_transform(soup)

cosine_sim = cosine_similarity(count_matrix)

# %% [markdown]
# We will reuse the get_recommendations function that we had written earlier. Since our cosine similarity scores have changed, we expect it to give us different (and probably better) results. Let us check for **The Dark Knight** again and see what recommendations I get this time around.

# %%
recommend('The Dark Knight').head(10)

# %% [markdown]
# I am much more satisfied with the results I get this time around. The recommendations seem to have recognized other Christopher Nolan movies (due to the high weightage given to director) and put them as top recommendations. I enjoyed watching **The Dark Knight** as well as some of the other ones in the list including **Batman Begins**, **The Prestige** and **The Dark Knight Rises**. 
# 
# We can of course experiment on this engine by trying out different weights for our features (directors, actors, genres), limiting the number of keywords that can be used in the soup, weighing genres based on their frequency, only showing movies with the same languages, etc.

# %% [markdown]
# Let me also get recommendations for another movie, **Mean Girls** which happens to be my girlfriend's favorite movie.

# %%
recommend('Mean Girls').head(10)

# %% [markdown]
# #### Ratings and Popularity
# 
# One thing that we notice about our recommendation system is that it recommends movies regardless of *ratings* and *popularity*.
# 
# Therefore, we will add a mechanism to reorder and return movies which are popular and have had a good critical response.
# 
# First I will take the top 25 movies based on similarity scores. Then we will calculate the weighted rating of each movie using IMDB's formula like we did in the Simple Recommender section. And using vote of the 60% as the value $m$ of these similar movies, then reorder.

# %%
def improved_recommendations(title):
    movies = recommend(title)[:25]
    md_s = pd.read_csv('../input/movies-data/metadata_small.csv', dtype=
    {'id': int, 'vote_count': int, 'vote_averages': float})
    md_s = md_s[md_s['id'].isin(movies.index)]
    return weighted_rating(md_s, 0.6)

# %%
improved_recommendations('The Dark Knight')

# %% [markdown]
# Let me also get the recommendations for **Mean Girls**, my girlfriend's favorite movie.

# %%
improved_recommendations('Mean Girls')

# %% [markdown]
# However, there is nothing much we can do about this. Therefore, we will conclude our Content Based Recommender section here and come back to it when we build a hybrid engine.

# %% [markdown]
# ## Collaborative Filtering
# 
# Our content based engine suffers from some severe limitations. It is only capable of suggesting movies which are *close* to a certain movie. That is, it is not capable of capturing tastes and providing recommendations across genres.
# 
# Also, the engine that we built is not really personal in that it doesn't capture the personal tastes and biases of a user. Anyone querying our engine for recommendations based on a movie will receive the same recommendations for that movie, regardless of who s/he is.
# 
# Therefore, in this section, we will use a technique called **Collaborative Filtering** to make recommendations which is based on the idea that users similar to me can be used to predict how much I will like a particular product or service those users have used/experienced but I have not.

# %%
from surprise import Reader, Dataset, SVD
from surprise import accuracy
from surprise.model_selection import train_test_split

ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')
ratings.head()

# %% [markdown]
# I will use the extremely powerful algorithms like **Singular Value Decomposition (SVD)** to minimise RMSE (Root Mean Square Error) and give great recommendations.

# %%
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], Reader())

# sample random trainset and testset
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=.25)

# We'll use the famous SVD algorithm.
algo = SVD()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)

# %% [markdown]
# We get a mean **Root Mean Sqaure Error** about 0.89 which is more than good enough for our case. Let us now train on our dataset and arrive at predictions.

# %%
trainset = data.build_full_trainset()
algo.fit(trainset)

# %% [markdown]
# Let us pick user 1 and check the ratings s/he has given.
# 
# 

# %%
algo.predict(1, 302)

# %% [markdown]
# For movie with ID 302, we get an estimated prediction of **2.8**. One startling feature of this recommender system is that it doesn't care what the movie is (or what it contains). It works purely on the basis of an assigned movie ID and tries to predict ratings based on how the other users have predicted the movie.

# %% [markdown]
# ## Hybrid Recommender
# 
# ![](https://www.toonpool.com/user/250/files/hybrid_20095.jpg)

# %% [markdown]
# In this section, I will try to build a simple hybrid recommender that brings together techniques we have implemented in the content based and collaborative filter based engines. This is how it will work:
# 
# * **Input:** User ID and the Title of a Movie
# * **Output:** Similar movies sorted on the basis of expected ratings by that particular user.

# %%
id_map = pd.read_csv('../input/the-movies-dataset/links_small.csv',
                     usecols=['movieId', 'tmdbId'])
id_map = id_map.dropna().astype(int).set_index('tmdbId')


def hybrid(userid, title):
    movies = recommend(title)
    movies['est'] = [algo.predict(userid, id_map.loc[x]['movieId']).est for x in movies.index]
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)

# %%
hybrid(1, 'Avatar')

# %%
hybrid(500, 'Avatar')

# %% [markdown]
# We see that for our hybrid recommender, we get different recommendations for different users although the movie is the same. Hence, our recommendations are more personalized and tailored towards particular users.

# %% [markdown]
# ## Conclusion
# 
# In this notebook, I have built 4 different recommendation engines based on different ideas and algorithms. They are as follows:
# 
# 1. **Simple Recommender:** This system used overall TMDB Vote Count and Vote Averages to build Top Movies Charts, in general and for a specific genre. The IMDB Weighted Rating System was used to calculate ratings on which the sorting was finally performed.
# 2. **Content Based Recommender:** We built two content based engines; one that took movie overview and taglines as input and the other which took metadata such as cast, crew, genre and keywords to come up with predictions. We also deviced a simple filter to give greater preference to movies with more votes and higher ratings.
# 3. **Collaborative Filtering:** We used the powerful Surprise Library to build a collaborative filter based on single value decomposition. The RMSE obtained was less than 1 and the engine gave estimated ratings for a given user and movie.
# 4. **Hybrid Engine:** We brought together ideas from content and collaborative filterting to build an engine that gave movie suggestions to a particular user based on the estimated ratings that it had internally calculated for that user.
# 
# Previous -> [The Story of Film](https://www.kaggle.com/rounakbanik/the-story-of-film/)
# 
# 
# 
# 


