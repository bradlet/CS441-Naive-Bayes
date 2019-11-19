# Bradley Thompson
# CS441 Artificial Intelligence - Bart Massey

from statistics import mean, stdev
from math import exp, sqrt, pi, log


# I see this as collapsing the list of cols down;
# result is a list of tuples that have the mean and st. dev. for each column.
def calc_col_stats(cols):
    col_stats = []
    for col in cols:
        col_stats.append((mean(col), stdev(col)))
    return col_stats


# Determine the probability that x is part of a feature
# https://en.wikipedia.org/wiki/Gaussian_function
def feature_probability(x, feature):
    mean, stdev = feature
    z_score = (x - mean) / stdev
    return (1/(stdev*sqrt(2*pi))) * exp(-(z_score**2)/2)
# NOTE TO SELF: MIGHT JUST HAVE TO REDO HOW I AM DOING PROBABILITY ENTIRELY HERE^^^^^^

def row_probability(row, data_set):
    class_prob = dict()
    for classification in data_set:
        class_prob[classification] = 1
        #for i in range(len(row)):
        #class_prob[classification] *= feature_probability(row[i],data_set[classification][i])
