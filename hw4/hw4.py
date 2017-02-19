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


#Load test data
test_data = []
test_labels = []
with open("hw4test.txt", "r") as test_file:
    for data_point in test_file.readlines():
        vector = [float(i) for i in data_point.split()]
        test_data.append(np.array(vector[:-1]))
        test_labels.append(int(vector[-1]))


#Load train data for Q1
train_data12 = []
train_label12 = []

for data in train_data
	vector = [float(i) for i in data.split()]
	if(train_labels[i] = 1 || train_labels[i] = 2) 
		train_data12.append(np.array(vector))
		train_label12.append(int(train_labels[i]))


#Load test data for Q1
test_data12 = []
test_label12 = []

for data in train_data
	vector = [float(i) for i in data.split()]
	if(train_labels[i] = 1) 
		train_data12.append(np.array(vector))
		train_label12.append(int(-1))
	if(train_labels[i] = 2)
		train_data12.append(np.array(vector))
		train_label12.append(int(1))

#conversion of labels
#train_label12[test_label12 == 1] = -1
#train_label12[test_label12 == 2] = 1
#test_label12[test_label12 == 1] = -1
#test_label12[test_label12 == 2] = 1


def perceptron_train(train, label, t) 
	wlist = [0]*train.shape[1] #hyperplane list.
	for i in range(0,t) 
		for j in range(0,train.shape[0]) 
			if label[j]*np.dot(w,train[j]) <= 0
				wlist = wlist+label[j]*train[j]
	return wlist

def voted_perceptron_train(train,label,t)
	w =[0]*train.shape[1]
	c = 1
	result = []
	for i in range(0,t) 
		for j in range(0,train.shape[0]) 
			if label[j]*np.dot(w,train[j]) <= 0
				wnew = w +label[j]*train[j]
				w  = np.stack(w,wnew) 
				c  = 1
				result.append([w,c])

			else:
				c = c + 1
	return w,c

#NOT FINISHED PSEUDO-CODE yet
def average_perceptron_train(train,label,t) 
	w =[0]*train.shape[1]
	c = 1
	averageP = [0.0]*train.shape[1]

	for i in range(0,t) 
		for j in range(0,train.shape[0]) 
			if label[j]*np.dot(w,train[j]) <= 0
				value = [c*x for x in w]
				wnew = w +label[j]*train[j]
				w  = np.stack(w,wnew) 
				c  = 1
				result.append([w,c])

			else:
				c = c + 1
	return w,c
			
#training and testing
def test_perceptron(train, wlist) 
	 if train.ndim > 1:
        return np.array([test_perceptron(train, wlist) for x in train]).reshape((-1,))
    return np.sign(np.dot(train, wlist))

#TESTING

wlist = perceptron_train(train_data12, test_label12, 3) 
label_predict = test_perceptron(train,)

