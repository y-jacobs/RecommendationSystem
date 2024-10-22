# Yahel Jacobs
# 313385536
import sys
import pandas as pd
import numpy as np
import heapq
from sklearn.metrics.pairwise import pairwise_distances


class collaborative_filtering:
    def __init__(self):
        self.ratings_table = []
        self.movies_subset = []
        self.user_based_matrix = []
        self.item_based_metrix = []
        self.users = []
        self.movies = []

    def create_fake_user(self, rating):
        "*** YOUR CODE HERE ***"
        # dicti = {"userId": 283238, "movieId": float('NaN'), "rating": float('NaN')}
        # df = pd.DataFrame(dicti)
        rating.loc[len(rating.index)] = [283238, float('NaN'), float('NaN')]
        # user = pd.DataFrame({"userId": [283238], "movieId": [pd.nan], "rating": [pd.nan]})
        # rating.append(user)
        return rating

    def keep_top_k(self, arr, k=10):
        smallest = heapq.nlargest(k, arr)[-1]
        arr[arr < smallest] = 0  # replace anything lower than the cut off with 0
        return arr

    def create_user_based_matrix(self, data):
        ratings = data[0]
        self.movies_subset = data[1]

        #for adding fake user
        ratings = self.create_fake_user(ratings)

        "*** YOUR CODE HERE ***"

        self.users = ratings.userId.unique().tolist()
        self.movies = ratings.movieId.unique().tolist()
        self.users.sort()
        self.movies.sort()

        table = (ratings.pivot(index='userId', columns='movieId', values='rating')).astype('float32')
        table = table.to_numpy(dtype='float32')
        self.ratings_table = table
        mean_user_rating = np.nanmean(table, axis=1).reshape(-1, 1)
        ratings_diff = (table - mean_user_rating)
        ratings_diff[np.isnan(ratings_diff)] = 0
        user_similarity = 1 - pairwise_distances(ratings_diff, metric='cosine')

        user_similarity = np.array([self.keep_top_k(np.array(arr)) for arr in user_similarity])
        self.user_based_matrix = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T

        # table = np.empty((len(users), len(movies)), dtype='float32')
        # table[:] = np.NaN

        # table = ratings.pivot(index='userId', columns='movieId', values='rating')
        # # np_table = table.to_numpy(dtype='float32')
        # mean_user_rating = table.mean(axis=1).to_numpy(dtype='float32').reshape(-1, 1)
        # mean_user_rating.round(2)
        # ratings_diff = (table - mean_user_rating)
        # ratings_diff[np.isnan(ratings_diff)] = 0
        # # ratings_diff.round(2)
        # user_similarity = 1-pairwise_distances(ratings_diff, metric='cosine')
        # self.user_based_matrix = pd.DataFrame(user_similarity.dot(ratings_diff).round(2))
        # print(0)
        # sys.exit(1)

    def create_item_based_matrix(self, data):
        "*** YOUR CODE HERE ***"
        ratings = data[0]
        self.movies_subset = data[1]

        self.users = ratings.userId.unique().tolist()
        self.movies = ratings.movieId.unique().tolist()
        self.users.sort()
        self.movies.sort()

        table = ratings.pivot(index='userId', columns='movieId', values='rating')
        table = table.to_numpy(dtype='float32')
        self.ratings_table = table

        mean_user_rating = np.nanmean(table, axis=1).reshape(-1, 1)
        ratings_diff = (table - mean_user_rating)
        ratings_diff[np.isnan(ratings_diff)] = 0

        item_similarity = 1-pairwise_distances(ratings_diff.T, metric='cosine')
        pd.DataFrame(item_similarity).round(2)

        k = 10
        item_similarity = np.array([self.keep_top_k(np.array(arr), k) for arr in item_similarity])
        self.item_based_metrix = (mean_user_rating + ratings_diff.dot(item_similarity) / np.array([np.abs(item_similarity).sum(axis=1)])).T


        # ratings = data[0]
        # mean_user_rating = ratings.mean(axis=1).to_numpy().reshape(-1, 1)
        # mean_user_rating.round(2)
        # ratings_diff = (ratings - mean_user_rating)
        #
        # ratings_diff[np.isnan(ratings_diff)] = 0
        # ratings_diff.round(2)
        #
        # raitingItem = ratings_diff
        # raitingItem[np.isnan(raitingItem)] = 0
        #
        # self.item_based_metrix = 1 - pairwise_distances(raitingItem.T, metric='cosine')

        # self.item_based_metrix = mean_user_rating + raitingItem.dot(item_similarity) / np.array([np.abs(item_similarity).sum(axis=1)])
        # sys.exit(1)

    def predict_movies(self, user_id, k, is_user_based=True):
        "*** YOUR CODE HERE ***"
        user_idx = self.users.index(int(user_id))
        data_matrix_row = self.ratings_table[user_idx]

        if is_user_based:
            predicted_ratings_row = self.user_based_matrix[user_idx]
        else:
            predicted_ratings_row = self.item_based_metrix[user_idx]

        predicted_ratings_unrated = predicted_ratings_row[np.isnan(data_matrix_row)]
        # print(predicted_ratings_unrated)

        idx = np.argsort(-predicted_ratings_unrated)
        sim_scores = idx[0:k]
        return self.movies_subset.title.iloc[sim_scores]