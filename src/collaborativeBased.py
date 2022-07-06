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
        ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')
        data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], Reader())

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
        self.algo.predict(1, 302)
