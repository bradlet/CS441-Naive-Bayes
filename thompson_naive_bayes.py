# Bradley Thompson
# CS441 Artificial Intelligence - Bart Massey

from supp.resplit import parse
from math import log

# Split the data by classification
def split_by_class(data):
    data_set = {}
    for (classification, features) in data:
        # if this classification hasn't been recognized, add it as a key to the data set
        if classification not in data_set:
            data_set[classification] = []
        # add the row of features to the list at that key in the data set dict
        data_set[classification].append(features)
    return data_set


# https://stackoverflow.com/questions/44360162/how-to-access-a-column-in-a-list-of-lists-in-python
# used to figure out how to get columns from a list of rows easily.
def get_cols(data):
    cols = list(zip(*data))
    return cols


# condense a column by getting the sum of the list
# result: an array with the # of times a feature appears positive
def positive_features(cols):
    return list(map(lambda x: sum(x), cols))


# Basically just the implementation for the learner pseudo-code in the assignment handout
def train(filename):
    data_set = split_by_class(parse(filename))
    counts = dict()
    for classification in data_set:
        class_count = len(data_set[classification])
        feature_counts = positive_features(get_cols(data_set[classification]))
        counts[classification] = (class_count, feature_counts) # split this up for readability, was one line
    return counts


# Compute likelihood an instance is of a given class
def probability_instance_of(instance, counts):
    prob = dict()
    print (counts)
    for i in range(len(counts)):
        n = log(counts[i][0] + 0.5)
        prob[i] = n - log(counts[0][0] + counts[1][0] + 0.5)
        for j in range(len(instance)):
            temp = counts[i][1][j]
            if (instance[j] == 0):
                temp = counts[i][0] - temp
            prob[i] = prob[i] + log(temp + 0.5) - n
    return prob


# guess classification based on likelihood returned from probability_instance_of
def classify(prob):
    if (prob[1] > prob[0]):
        return 1
    return 0


# positive = normal, negative = abnormal
def run(test):
    train_set = train("supp/spect-" + test + ".train.csv")
    test_set = parse("supp/spect-" + test+ ".test.csv")
    positive = negative = pos_correct = neg_correct = 0
    for instance in test_set:
        if classify(probability_instance_of(instance[1],train_set)) is 1:
            positive += 1
            if instance[0] is 1:
                pos_correct += 1
        else:
            negative += 1
            if instance[0] is 0:
                neg_correct += 1
    total = positive + negative
    correct = pos_correct + neg_correct
    print (test, " ",
        correct, "/", total, "(", round((correct/total), 2), ") ",
        neg_correct, "/", negative, "(", round((neg_correct/negative), 2), ") ",
        pos_correct, "/", positive, "(", round((pos_correct/positive), 2), ") ", sep="")


for test in ["itg", "orig", "resplit"]:
    run(test)

