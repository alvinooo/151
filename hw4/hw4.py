import sys
import numpy as np

#Load train data
train_data = []
train_labels = []
with open("hw4train.txt", "r") as train_file:
    for data_point in train_file.readlines():
        vector = [float(i) for i in data_point.split()]
        train_data.append(np.array(vector[:-1]))
        train_labels.append(int(vector[-1]))

test_data = []
test_labels = []
with open("hw4test.txt", "r") as test_file:
    for data_point in test_file.readlines():
        vector = [float(i) for i in data_point.split()]
        test_data.append(np.array(vector[:-1]))
        test_labels.append(int(vector[-1]))

train_12 = [pair for pair in filter(lambda pair:pair[1] == 1 or pair[1] == 2, zip(train_data,train_labels))]
train_data12, train_label12 = zip(*train_12)
train_label12 =[x for x in map(lambda x:1 if x == 1 else -1,train_label12)]

test_12 = [pair for pair in filter(lambda pair:pair[1] == 1 or pair[1] == 2, zip(test_data,test_labels))]
test_data12, test_label12 = zip(*test_12)
test_label12 =[x for x in map(lambda x:1 if x == 1 else -1,test_label12)]

def perceptron(train, label, t):
    wlist = [0]*len(train[0]) #hyperplane list.
    for i in range(0,t):
        for data, label2 in zip(train,label):
            if label2*np.dot(wlist,data) <= 0:
                wlist = wlist+label2*data
    return wlist

def error(data,label,wlist):
    error = 0
    for d,l in zip(data,label):
        if np.dot(wlist,d)*l <= 0:
            error = error + 1
    return float(error)/len (label)

def voted_perceptron_train(train,label,t):
    w =np.array([0]*len(train[0]))
    c = 1
    result = []
    for i in range(0,t):
        for data, label2 in zip(train,label):
            if label2*np.dot(w,data) <= 0:
                result.append([w,c])
                w = w +label2*data
                c  = 1
            else:
                c = c + 1
    return result

def error_voted(data, label, weights, counts):
    error = 0
    for d, l in zip(data, label):
        if np.dot(counts, np.sign(np.dot(weights, d))) * l <= 0:
            error += 1
    return float(error) / len(label)

def error_average(data,label,weights,counts):
    error = 0
    for d,l in zip(data,label):
        value = sum([w*c for w,c in zip(weights,counts)])
        if (np.dot(value,d)) * l <= 0:
            error += 1
    return float (error)/ len(label)

print("PERCEPTRON")
for i in range(1, 5):
    print "Pass",i
    weights = perceptron(train_data12,train_label12,i)
    print "Training Error ", error(train_data12, train_label12, weights)
    print "Test Error ", error(test_data12, test_label12, weights)

print("VOTED PERCEPTRON")
for i in range (1,5):
    print "Pass",i
    weights, counts = zip(*voted_perceptron_train(train_data12,train_label12,i))
    print "Training Error ", error_voted(train_data12, train_label12, weights,counts)
    print "Test Error ", error_voted(test_data12, test_label12, weights,counts)

print("VOTED AVERAGE")
for i in range (1,5):
    print "Pass", i
    weights, counts = zip(*voted_perceptron_train(train_data12,train_label12,i))
    print "Training Error ", error_average(train_data12, train_label12, weights,counts)
    print "Test Error ", error_average(test_data12, test_label12, weights,counts)

weights, counts = zip(*voted_perceptron_train(train_data12, train_label12, 3))
avg_weight = sum([w*c for w,c in zip(weights,counts)])

dictionary = []
with open("hw4dictionary.txt", "r") as d:
    for word in d.readlines():
        dictionary.append(word.strip())

print "Positive"
for i in np.argsort(avg_weight)[-3:]:
    print dictionary[i]

print "Negative"
for i in np.argsort(avg_weight)[::-1][-3:]:
    print dictionary[i]

train_labels_class = [[x for x in map(lambda label:1 if label == clas else -1,train_labels)] for clas in xrange(1, 7)]
weights = [perceptron(train_data, train_labels_class[i], 1) for i in xrange(6)]

confusion = [np.array([0.0 for _ in xrange(7)]) for _ in xrange(6)]
IDK = 6
for test, label in zip(test_data, test_labels):
    predictions = [np.sign(np.dot(weight, test)) for weight in weights]
    if len(filter(lambda x: x == 1, predictions)) != 1:
        confusion[label - 1][IDK] += 1
    else:
        confusion[label - 1][np.argmax(predictions)] += 1

for row in confusion:
    print row
for label in confusion:
    label /= float(sum(label))
for row in confusion:
    print row