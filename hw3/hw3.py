import sys
import numpy as np
from collections import deque, Counter

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

features = {}
with open("hw3features.txt", "r") as feature_file:
    index = 0
    for feature in feature_file.readlines():
        features[index] = feature.strip()
        index += 1

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

def compute_entropy(distributions):
    total = float(sum([size[0] for size in distributions]))
    entropy = 0
    for size, distribution in distributions:
        if size > 0:
            entropy -= np.nan_to_num(np.dot(distribution / float(size), np.log(distribution / float(size)))) * size
    return entropy / total

class Node:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.majority_label = None
        self.label = None
        self.left = None
        self.right = None

class ID3Tree:
    def __init__(self, data, labels):
        self.root = self.build_tree(data, labels)
    
    def build_tree(self, data, labels):
        node = Node()
        distinct_labels = list(set(labels))
        
        # Pure node
        if len(distinct_labels) == 1:
            node.label = distinct_labels[0]
            return node
        
        # Majority label
        node.majority_label = Counter(labels).most_common()[0][0]
        
        # Recursively split data
        feature, threshold = split(data, labels)
        feature_sort = sorted(zip(data, labels), key=lambda point: point[0][feature])
        left_data, left_labels = zip(*filter(lambda x: x[0][feature] < threshold, feature_sort))
        right_data, right_labels = zip(*filter(lambda x: x[0][feature] >= threshold, feature_sort))
        
        node.feature = feature
        node.threshold = threshold
        node.left = self.build_tree(left_data, left_labels)
        node.right = self.build_tree(right_data, right_labels)

        return node

    def classify(self, root, data_point, prune_node=None):
        if prune_node and root == prune_node:
            return root.majority_label
        if root.left and data_point[root.feature] < root.threshold:
            return self.classify(root.left, data_point, prune_node)
        elif root.right and data_point[root.feature] >= root.threshold:
            return self.classify(root.right, data_point, prune_node)
        else:
            return root.label
    
    def accuracy(self, data, labels, prune_node=None):
        correct = 0.0
        for point, label in zip(data, labels):
            if self.classify(self.root, point, prune_node) == label:
                correct += 1
        return correct / len(data)
    
    def prune(self, data, labels):
        bfs_queue = deque([self.root])
        while len(bfs_queue) > 0:
            curr = bfs_queue.popleft()
            original, pruned = self.accuracy(data, labels), self.accuracy(data, labels, curr)
            if pruned > original:
                curr.label = curr.majority_label
                curr.left, curr.right = None, None
                return
            if curr.left:
                bfs_queue.append(curr.left)
            if curr.right:
                bfs_queue.append(curr.right)

    def bfs(self):
        bfs_queue = deque([self.root])
        while len(bfs_queue) > 0:
            curr = bfs_queue.popleft()
            if curr.left or curr.right:
                print features[curr.feature], "<", curr.threshold, "majority", curr.majority_label
            else:
                print "Prediction", curr.label
            if curr.left:
                bfs_queue.append(curr.left)
            if curr.right:
                bfs_queue.append(curr.right)

tree = ID3Tree(train_data, train_labels)
tree.bfs()

print "Training accuracy =", tree.accuracy(train_data, train_labels)
print "Test accuracy =", tree.accuracy(test_data, test_labels)

tree.prune(validate_data, validate_labels)
print "Round 1 of pruning"
print "Validation accuracy =", tree.accuracy(validate_data, validate_labels)
print "Test accuracy =", tree.accuracy(test_data, test_labels)
# tree.bfs()

tree.prune(validate_data, validate_labels)
print "Round 2 of pruning"
print "Validation accuracy =", tree.accuracy(validate_data, validate_labels)
print "Test accuracy =", tree.accuracy(test_data, test_labels)

tree.bfs()