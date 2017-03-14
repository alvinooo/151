import numpy as np
import math as math

train_data = []
train_labels = []
test_data = []
test_labels = []

with open("hw6train.txt") as train_file:
    for data_point in train_file.readlines():
        vector = [int(i) for i in data_point.split()]
        train_data.append(np.array(vector[:-1]))
        train_labels.append(int(vector[-1]))

with open("hw6test.txt") as train_file:
    for data_point in train_file.readlines():
        vector = [int(i) for i in data_point.split()]
        train_data.append(np.array(vector[:-1]))
        train_labels.append(int(vector[-1]))

def weak_pos_learner(value):
    return 1 if value == 1 else -1

def weak_neg_learner(value):
    return 1 if value == 0 else -1   


def alpha(err):
    return 0.5*math.log((1.0-err)/err)

def classifier_error(data,labels):
    leng = len(data[0])
    errors = [0 for i in range(0,2*leng)]
    for coordinates in range(0,leng):
        for email in range(0,len(data)):
            if (weak_neg_learner(data[email][coordinates]) != labels[email]):
                errors[coordinates+leng] += weights[email] 
            if(weak_pos_learner(data [email][coordinates]) != labels[email]):
                errors[coordinates] += weights[email]
    return errors

def classifier(error_ls):
    err  = 9999.999
    for i in range(0,len(error_ls)):
        if (error_ls[i] < err):
            err =  float(error_ls[i])
            feature = i
    return (feature,err)


def classifier(error_ls):
    err  = 9999.999
    for i in range(0,len(error_ls)):
        if (error_ls[i] < err):
            err =  float(error_ls[i])
            feature = i
    return (feature,err)


def boosting(t,data,label):
    output = []
    leng = len(data[0])
    for i in range (0,t):
        classifiers = classifier_error(data,label)
        feature_ht = classifier(classifiers)[0]
        min_er = classifier(classifiers)[1]
        alfa = alpha(min_er)
        value = 0
        for j in range(0, len(weights)):
            if(feature_ht < leng):
                value = math.exp(-alfa*label[j]*weak_pos_learner(data[j][feature_ht]))
                weights[j] = (weights[j]*value)
            if(feature_ht >= leng):
                value = math.exp(-alfa*label[j]*weak_neg_learner(data[j][feature_ht-leng]))
                weights[j] = (weights[j]*value)
        s = sum(weights)
        for j in range(0, len(weights)):
            weights[j] = weights[j]/s
        output.append((feature_ht,alfa))
    return output

weights = initialize()
output = boosting(4,train_data,train_labels)

def final_classifier(list_classifiers, email):
    sum = 0
    for i in range(0,len(list_classifiers)):
        alpha = list_classifiers[i][1]
        feature = list_classifiers[i][0]
        if(feature < len(train_data[0])):
            h = weak_pos_learner(email[feature])
        else:
            h = weak_neg_learner(email[feature-len(train_data[0])])
        sum += alpha*h
    return int(np.sign(sum))

def set_errors(data,label,classifiers):
    error = 0
    for i in range(0,len(data)):
        if (final_classifier(classifiers,data[i]) != label[i]):
            error += 1
    return float(error)/ float(len(data))
            
set_errors(train_data,train_labels, output)