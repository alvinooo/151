from collections import defaultdict

train_data = []
train_labels = []
with open("hw5train.txt", "r") as train_file:
    for data_point in train_file.readlines():
        train_data.append(data_point.split()[0])
        train_labels.append(int(data_point.split()[1]))

test_data = []
test_labels = []
with open("hw5test.txt", "r") as test_file:
    for data_point in test_file.readlines():
        test_data.append(data_point.split()[0])
        test_labels.append(int(data_point.split()[1]))

def kernel_multi_count(s, t, p):
    s_sub, t_sub = defaultdict(int), defaultdict(int)
    for i in xrange(0, len(s) - p + 1):
        s_sub[s[i:i + p]] += 1
    for i in xrange(0, len(t) - p + 1):
        t_sub[t[i:i + p]] += 1
    return sum([s_sub[sub] * t_sub[sub] for sub in s_sub])

def kernel_single_count(s, t, p):
    s_sub, t_sub = defaultdict(int), defaultdict(int)
    for i in xrange(0, len(s) - p + 1):
        s_sub[s[i:i + p]] = 1
    for i in xrange(0, len(t) - p + 1):
        t_sub[t[i:i + p]] = 1
    return sum([s_sub[sub] * t_sub[sub] for sub in s_sub])

def predict(mistakes, k, x, p):
    return sum([my * k(mx, x, p) for mx, my in mistakes])

def kernel_perceptron(train, labels, k, p):
    mistakes = []
    for data, label in zip(train, labels):
        if predict(mistakes, k, data, p) * label <= 0:
            mistakes.append((data, label))
    return mistakes

def error(mistakes, data, labels, k, p):
    wrong = 0
    for d, l in zip(data, labels):
        if predict(mistakes, k, d, p) * l <= 0:
            wrong += 1
    return float(wrong) / len(data)

def string_weights(strings, mistakes, k, p):
    weights = {}
    for s in strings:
        weights[s] = predict(mistakes, k, s, p)
    return weights

# mistakes_multi_3 = kernel_perceptron(train_data, train_labels, kernel_multi_count, 3)
# mistakes_multi_4 = kernel_perceptron(train_data, train_labels, kernel_multi_count, 4)
# mistakes_multi_5 = kernel_perceptron(train_data, train_labels, kernel_multi_count, 5)
# print error(mistakes_multi_3, train_data, train_labels, kernel_multi_count, 3)
# print error(mistakes_multi_4, train_data, train_labels, kernel_multi_count, 4)
# print error(mistakes_multi_5, train_data, train_labels, kernel_multi_count, 5)
# print error(mistakes_multi_3, test_data, test_labels, kernel_multi_count, 3)
# print error(mistakes_multi_4, test_data, test_labels, kernel_multi_count, 4)
# print error(mistakes_multi_5, test_data, test_labels, kernel_multi_count, 5)

# mistakes_single_3 = kernel_perceptron(train_data, train_labels, kernel_single_count, 3)
# mistakes_single_4 = kernel_perceptron(train_data, train_labels, kernel_single_count, 4)
# mistakes_single_5 = kernel_perceptron(train_data, train_labels, kernel_single_count, 5)
# print error(mistakes_single_3, train_data, train_labels, kernel_single_count, 3)
# print error(mistakes_single_4, train_data, train_labels, kernel_single_count, 4)
# print error(mistakes_single_5, train_data, train_labels, kernel_single_count, 5)
# print error(mistakes_single_3, test_data, test_labels, kernel_single_count, 3)
# print error(mistakes_single_4, test_data, test_labels, kernel_single_count, 4)
# print error(mistakes_single_5, test_data, test_labels, kernel_single_count, 5)

mistakes_multi_5 = eval(open("mistakes.txt", "r").read())
mistakes_strings = [pair[0] for pair in mistakes_multi_5]

substrings = set()
for s in mistakes_strings:
    for i in range(0, len(s) - 4):
        substrings.add(s[i:i+5])

weights = string_weights(substrings[10000:20000], mistakes_multi_5, kernel_multi_count, 5)
open("positive_weights_10000.txt", "w").write(str(weights))

weights = string_weights(substrings[20000:30000], mistakes_multi_5, kernel_multi_count, 5)
open("positive_weights_20000.txt", "w").write(str(weights))

weights = string_weights(substrings[30000:40000], mistakes_multi_5, kernel_multi_count, 5)
open("positive_weights_30000.txt", "w").write(str(weights))

weights = string_weights(substrings[40000:50000], mistakes_multi_5, kernel_multi_count, 5)
open("positive_weights_40000.txt", "w").write(str(weights))

weights = string_weights(substrings[50000:], mistakes_multi_5, kernel_multi_count, 5)
open("positive_weights_50000.txt", "w").write(str(weights))