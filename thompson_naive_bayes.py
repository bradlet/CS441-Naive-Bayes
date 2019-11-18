# Bradley Thompson
# CS441 Artificial Intelligence - Bart Massey

from statistics import mean, stdev
from supp.resplit import parse

""" Steps:
1. load in data
2. split by classification
3. get mean and stdev of each column in the features
    - Need this for each column...
4. calc probability 
Note:
    P(A|B) = P(B|A)P(A) / P(B)
    Naive = Assume mutually exclusive:
        P(B|A1) P(B|A2)...P(B|An) P(A)

5. based on probability, guess class 
6. accuracy measures
"""


# https://stackoverflow.com/questions/44360162/how-to-access-a-column-in-a-list-of-lists-in-python
# used to figure out how to get columns from a list of rows easily.
def get_cols(data):
    cols = list(zip(*data))
    return cols


# I see this as collapsing the list of cols down;
# result is a list of tuples that have the mean and st. dev. for each column.
def calc_col_stats(cols):
    col_stats = []
    for col in cols:
        col_stats.append((mean(col), stdev(col)))
    return col_stats


# split the data by classification
def classify(data):
    data_set = {}
    for (classification, features) in data:
        # if this classification hasn't been recognized, add it as a key to the data set
        if classification not in data_set:
            data_set[classification] = []
        # add the row of features to the list at that key in the data set dict
        data_set[classification].append(features)
    return data_set


def run(filename):
    data = parse(filename)
    data_set = classify(data)
    for classification in data_set:
        data_set[classification] = get_cols(data_set[classification])
        data_set[classification] = calc_col_stats(data_set[classification])
    print(data_set)

run("supp/SPECT.train")
