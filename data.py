import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def watch_data_info(data):
    for d in data:
        # This function returns the first 5 rows for the object based on position.
        # It is useful for quickly testing if your object has the right type of data in it.
        print(d.head())

        # This method prints information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.
        print(d.info())

        # Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
        print(d.describe(include='all').transpose())


def print_data(data):
    "*** YOUR CODE HERE ***"
    d1 = data[0]
    print("unique users count: " + str(d1.userId.nunique()))
    print("unique movies count: " + str(d1.movieId.nunique()))
    print("rating count: " + str(len(d1)))
    print("count of most rated movie: " + str(d1.movieId.value_counts().max()))
    print("count of least rated movie: " + str(d1.movieId.value_counts().min()))
    print("count of most rating user: " + str(d1.userId.value_counts().max()))
    print("count of least rating user: " + str(d1.userId.value_counts().min()))


def plot_data(data, plot = True):
    "*** YOUR CODE HERE ***"
    if plot:
        d1 = data[0]
        d1.hist(column='rating')
        plt.show()


# def test(data):
#     d1 = data[0]
#
#     sim_scores = np.array([14, 17, 25, 62, 105,147, 175,193,222,249])
#
#     print(data[1].genres.iloc[sim_scores])