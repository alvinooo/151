import numpy as np
import math as math

train_data = []
train_labels = []
test_data = []
test_labels = []
dictionary = []

with open("hw6train.txt") as train_file:
    for data_point in train_file.readlines():
        vector = [int(i) for i in data_point.split()]
        train_data.append(np.array(vector[:-1]))
        train_labels.append(int(vector[-1]))

with open("hw6test.txt") as train_file:
    for data_point in train_file.readlines():
        vector = [int(i) for i in data_point.split()]
        test_data.append(np.array(vector[:-1]))
        test_labels.append(int(vector[-1]))
    
with open("dictionary.txt") as dic_file:
    for data_point in dic_file.readlines():
        dictionary.append(data_point)

def weak_pos_learner(value):
    return 1 if value == 1 else -1

def weak_neg_learner(value):
    return 1 if value == 0 else -1    

def alpha(err):
    return 0.5*math.log((1-err)/err)

def classifier_error(data,labels):
    leng = len(data[0])
    errors = [0 for i in range(0,2*leng)]
    for coordinates in range(0,leng):
        for email in range(0,len(data)):
            if (weak_neg_learner(data[email][coordinates]) != labels[email]):
                errors[coordinates+leng] += weights[email] 
            if(weak_pos_learner(data[email][coordinates]) != labels[email]):
                errors[coordinates] += weights[email]
    return errors

def classifier(error_ls):
    err  = 999999999
    for i in range(0,len(error_ls)):
        if (error_ls[i] < err):
            err =  float(error_ls[i])
            feature = i
    return (feature,err)

def initialize(data):
    weights = []
    w = float(1.0) / (len(data))
    for i in range (0, len(data)):
        weights.append(float(w))
    return weights

def boosting(t,data,label,weight):
    output = []
    leng = len(data[0])
    for i in range (0,t):
        classifiers = classifier_error(data,label)
        feature_ht = classifier(classifiers)[0]
        min_er = classifier(classifiers)[1]
        alfa = alpha(min_er)
        value = 0
        for j in range(0, len(weight)):
            if(feature_ht < leng):
                value = math.exp(-alfa*label[j]*weak_pos_learner(data[j][feature_ht]))
                weight[j] = (weight[j]*value)
            if(feature_ht >= leng):
                value = math.exp(-alfa*label[j]*weak_neg_learner(data[j][feature_ht-leng]))
                weight[j] = (weight[j]*value)
        w = sum(weight)
        for i in range(0,len(weight)):
            weight[i] = weight[i]/w
        output.append((feature_ht,alfa))
    return output

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
    return np.sign(sum)

def set_errors(data,label,classifiers):
    error = 0
    for i in range(0,len(data)):
        if (final_classifier(classifiers,data[i])!= label[i]):
            error += 1
    return float(error)/len(data)

#Checking training error for t = 4, = 0.051
weights = initialize(train_data)
output = boosting(4,train_data,train_labels,weights)
set_errors(train_data,train_labels, output)

#Checking testing error for t=4, = 0.038759
test_weights = initialize(test_data)
test_output = boosting(4,test_data,test_labels,test_weights)
set_errors(test_data,test_labels, output)

#PART 1 

#TRAINING SET 3,7,10,15,20
weights = initialize(train_data)
output = []
output.append(boosting(3,train_data,train_labels,weights))
output.append(boosting(7,train_data,train_labels,weights))
output.append(boosting(10,train_data,train_labels,weights))
output.append(boosting(15,train_data,train_labels,weights))
output.append(boosting(20,train_data,train_labels,weights))
for i in range(0,len(output)):
    print("Training error for round = ", i+1)
    print(set_errors(train_data,train_labels, output[i]))


#TEST SET 3,7,10,15,20
test_weights = initialize(test_data)
test_output = []
test_output.append(boosting(3,test_data,test_labels,test_weights))
test_output.append(boosting(7,test_data,test_labels,test_weights))
test_output.append(boosting(10,test_data,test_labels,test_weights))
test_output.append(boosting(15,test_data,test_labels,test_weights))
test_output.append(boosting(20,test_data,test_labels,test_weights))
for i in range(0,len(test_output)):
    print("Test error for round = ", i+1)
    print(set_errors(test_data,test_labels, output[i]))

#PART 2
dictionary_output_10 = (output[2])
print(dictionary_output_10)
for i in range(0,10):
    feature = dictionary_output_10[i][0]
    if (feature < 4003):
        print("Present of word : " , dictionary[feature])
    else:
        print("Absence of word : " , dictionary[int(feature/2)])
