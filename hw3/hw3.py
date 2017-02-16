import sys
import numpy as np
from collections import deque

class Node:
    def __init__(feature=None, threshold=None, majority_label=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.majority_label = majority_label
        self.label = label
        self.left = None
        self.right = None

train_data = []
train_labels = []
with open("hw3train.txt", "r") as train_file:
    for data_point in train_file.readlines():
        vector = [float(i) for i in data_point.split()]
        train_data.append(np.array(vector[:-1]))
        train_labels.append(int(vector[-1]))

validate_data = []
validate_labels = []
with open("hw3validation.txt", "r") as validate_file:
    for data_point in validate_file.readlines():
        vector = [float(i) for i in data_point.split()]
        validate_data.append(np.array(vector[:-1]))
        validate_labels.append(int(vector[-1]))

test_data = []
test_labels = []
with open("hw3test.txt", "r") as test_file:
    for data_point in test_file.readlines():
        vector = [float(i) for i in data_point.split()]
        test_data.append(np.array(vector[:-1]))
        test_labels.append(int(vector[-1]))

def compute_entropy(distributions):
    total = float(sum([size[0] for size in distributions]))
    entropy = 0
    for size, distribution in distributions:
        if size > 0:
            entropy -= np.nan_to_num(np.dot(distribution / float(size), np.log(distribution / float(size)))) * size
    return entropy / total

def best_threshold(feature_sort, feature):
    data, labels = zip(*feature_sort)
    
    # Find all possible splits
    distinct = sorted(set([point[feature] for point in data]))
    thresholds = [(low + hi) / 2.0 for low, hi in zip(distinct[:-1], distinct[1:])]
    
    # Conditional distributions
    label_counts = [len(filter(lambda x: x == 0, labels)), len(filter(lambda x: x == 1, labels))]
    low_counts, hi_counts = np.array([0, 0]), np.array(label_counts)
    
    # Find best threshold and entropy
    threshold, entropy = 0, compute_entropy([(0, low_counts), (len(data), hi_counts)])
    curr_threshold = 0
    for i in xrange(len(data)):
        
        if curr_threshold >= len(thresholds):
            break

        if data[i][feature] > thresholds[curr_threshold]:
            curr_entropy = compute_entropy([(i + 1, low_counts), (len(data) - i - 1, hi_counts)])
            if curr_entropy < entropy:
                entropy = curr_entropy
                threshold = thresholds[curr_threshold]
            curr_threshold += 1

        low_counts[labels[i]] += 1
        hi_counts[labels[i]] -= 1

    return threshold, entropy

def split(data, labels):
    feature, threshold, entropy = 0, 0, sys.maxint
#     print "Feature\tThreshold\tEntropy"
    for curr_feature in xrange(len(data[0])):
        feature_sort = sorted(zip(data, labels), key=lambda point: point[0][curr_feature])
        curr_threshold, curr_entropy = best_threshold(feature_sort, curr_feature)
#         print curr_feature, '\t', curr_threshold, '\t', curr_entropy
        if curr_entropy < entropy:
            feature, threshold, entropy = curr_feature, curr_threshold, curr_entropy
    return feature, threshold

def build_tree(data, labels):
    feature, threshold = split(data, labels)
    feature_sort = sorted(zip(data, labels), key=lambda point: point[0][feature])
    left_data, left_labels = zip(*filter(lambda x: x[0][feature] < threshold, feature_sort))
    right_data, right_labels = zip(*filter(lambda x: x[0][feature] >= threshold, feature_sort))
    print len(left_data), len(right_data) # TODO: Confirm by printing number of elements < 0.5
    # TODO output node
    # TODO recurse on left and right datasets
    # return node

    if labels.isPure():
    	return Node (None, None, None, labels[0])

    left = build_tree(left_data, left_labels)
    right = build_tree(right_data, right_labels)
    return Node(feature, threshold, labels.getMajority(), None)



build_tree(train_data, train_labels)