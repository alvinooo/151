import sys
import heapq
import numpy as np

train_file = open("hw2train.txt", "r")
train_data = []
train_labels = []
for data_point in train_file.readlines():
    vector = [int(i) for i in data_point.split()]
    train_data.append(np.array(vector[:-1]))
    train_labels.append(vector[-1])

validate_file = open("hw2validate.txt", "r")
validate_data = []
validate_labels = []
for data_point in validate_file.readlines():
    vector = [int(i) for i in data_point.split()]
    validate_data.append(np.array(vector[:-1]))
    validate_labels.append(vector[-1])

test_file = open("hw2test.txt", "r")
test_data = []
test_labels = []
for data_point in test_file.readlines():
    vector = [int(i) for i in data_point.split()]
    test_data.append(np.array(vector[:-1]))
    test_labels.append(vector[-1])

k = 1
train = zip(train_data, train_labels)

train_mistakes = 0
for train_data_point, label in train:
    heap = []
    for neighbor, neighbor_label in train:
        diff = train_data_point - neighbor
        heap.append((np.dot(diff, diff), neighbor_label))
    nearest_neighbors = heapq.nsmallest(k, heap)

    if label != nearest_neighbors[0][1]:
        train_mistakes += 1

print float(train_mistakes) / len(train_data)