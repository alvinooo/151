import sys
import heapq
import numpy as np
import random

train_data = []
train_labels = []
with open("hw2train.txt", "r") as train_file:
    for data_point in train_file.readlines():
        vector = [int(i) for i in data_point.split()]
        train_data.append(np.array(vector[:-1]))
        train_labels.append(vector[-1])

validate_data = []
validate_labels = []
with open("hw2validate.txt", "r") as validate_file:
    for data_point in validate_file.readlines():
        vector = [int(i) for i in data_point.split()]
        validate_data.append(np.array(vector[:-1]))
        validate_labels.append(vector[-1])

test_data = []
test_labels = []
with open("hw2test.txt", "r") as test_file:
    for data_point in test_file.readlines():
        vector = [int(i) for i in data_point.split()]
        test_data.append(np.array(vector[:-1]))
        test_labels.append(vector[-1])

project_matrix = []
with open("projection.txt", "r") as projection_file:
    for row in projection_file.readlines():
        project_matrix.append([float(i) for i in row.split()])

def error_rate(train, data, k):
    errors = 0
    for data_point, label in data:
        heap = []
        for neighbor, neighbor_label in train:
            diff = data_point - neighbor
            heapq.heappush(heap, (np.dot(diff, diff), random.random(), neighbor_label))
        nearest_neighbors = heapq.nsmallest(k, heap)

        counts = {}
        majority_label, count = nearest_neighbors[0][2], 1
        for _, _, neighbor_label in nearest_neighbors:
            if neighbor_label not in counts:
                counts[neighbor_label] = 0
            counts[neighbor_label] += 1
            if counts[neighbor_label] > count:
                majority_label, count = neighbor_label, counts[neighbor_label]

        if label != majority_label:
            errors += 1

    return float(errors) / len(data)

train = zip(train_data, train_labels)

print error_rate(train, zip(train_data, train_labels), 3)
for k in [1, 5, 9, 15]:
    print "k = ", k
    print "training error", error_rate(train, zip(train_data, train_labels), k)
    print "validation error", error_rate(train, zip(validate_data, validate_labels), k)
    print "test error", error_rate(train, zip(test_data, test_labels), k)
    print

train_projection = np.dot(np.array(train_data), np.array(project_matrix))
validate_projection = np.dot(np.array(validate_data), np.array(project_matrix))
test_projection = np.dot(np.array(test_data), np.array(project_matrix))

train = zip(train_projection, train_labels)

print error_rate(train, zip(train_projection, train_labels), 3)
for k in [1, 5, 9, 15]:
    print "k = ", k
    print "training error", error_rate(train, zip(train_projection, train_labels), k)
    print "validation error", error_rate(train, zip(validate_projection, validate_labels), k)
    print "test error", error_rate(train, zip(test_projection, test_labels), k)
    print