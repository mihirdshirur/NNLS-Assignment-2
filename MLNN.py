import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


df = pd.read_csv("/Users/apple/Desktop/Coding/Sem5/Assignment2_Code/iris.data")


data_setosa = df.iloc[0:50,0:4].values
data_versicolor = df.iloc[50:100,0:4].values
data_virginica = df.iloc[100:150,0:4].values

# Using Backpropogation Algorithm to classify 
# Setosa is of class 1, versicolor is of class 2, virginica is of class 3
data_class_setosa = np.concatenate((np.ones((50,1)),data_setosa,np.ones((50,1))),axis = 1)
data_class_versicolor = np.concatenate((np.ones((50,1)),data_versicolor,np.ones((50,1))*2),axis = 1)
data_class_virginica = np.concatenate((np.ones((50,1)),data_virginica,np.ones((50,1))*3),axis = 1)
data_class = np.concatenate((data_class_setosa,data_class_versicolor,data_class_virginica),axis = 0)
np.random.shuffle(data_class)
percentage = 0.7                # Percentage of training set
training_set = data_class[0:int(percentage*150),:]       # Training data
test_set = data_class[int(percentage*150):150,:]         # Test data
# Our multilayer neural network has
# 1) Four input nodes
# 2) 1 hidden layer with 8 neurons and 1 bias term
# 3) Three output nodes


# TRAIN MODEL 
def train_model(lr,alpha):
    w_12 = []   
    w_23 = []
                                                # learning rate
                                            # alpha
    e = np.ones((1,3))                                    # Error
    w_12.append(np.random.rand(5,8))                          # Weight matrix between layer 1 and 2 (5x2)
    w_23.append(np.random.rand(9,3))                          # Weight matrix between layer 2 and 3 (3x3)
    count = 0
    error = []
    for t in range(100):   
        np.random.shuffle(training_set)
        for i in range(int(percentage*150)):

            # Forward computation
            # Calculating output
            x_1 = training_set[i:i+1,0:5]                         # Input in layer 1 (1x5)
            v_2 = np.dot(x_1,w_12[count])                              # Induced local field in layer 2 (1x8)
            v_2 = np.concatenate((np.ones((1,1)),v_2),axis=1)       # Induced local field in layer 2 with bias (1x9)
            y_2 = 1/(1+np.exp((-1)*v_2))                        # Output in layer 2 (1x9)
            v_3 = np.dot(y_2,w_23[count])                              # Induced local field in layer 3 (1x3)
            y_3 = 1/(1+np.exp((-1)*v_3))                        # Output in layer 3 (1x3)        
            # Compute error signal
        
            if int(training_set[i,5]) == 1:
                d = np.array([[1,0,0]])                          # Desired signal (1x3)
            elif int(training_set[i,5]) == 2:
                d = np.array([[0,1,0]])                          # Desired signal (1x3)
            elif int(training_set[i,5]) == 3:           
                d = np.array([[0,0,1]])                          # Desired signal (1x3)
            else:
                print("Error!") 
            e = d - y_3                                         # Error in layer 3 (1x3)
            # Backward computation
            delta_3 = e * y_3 * (1 - y_3)                       # Delta in layer 3 (1x3)
            delta_2 = y_2 * (1 - y_2) * np.transpose((np.dot(w_23[count],np.transpose(delta_3))))  # Delta in layer 2 (1x9)
            # Adjust synaptic weights
            if i == 0:
                w_12_new = w_12[count] + lr * np.dot(np.transpose(x_1),delta_2[0:1,1:9])
                w_12.append(w_12_new)
                w_23_new = w_23[count] + lr * np.dot(np.transpose(y_2),delta_3)
                w_23.append(w_23_new)
            else:
                w_12_new = w_12[count] + lr * np.dot(np.transpose(x_1),delta_2[0:1,1:9]) + alpha * (w_12[count]-w_12[count-1])
                w_12.append(w_12_new)
                w_23_new = w_23[count] + lr * np.dot(np.transpose(y_2),delta_3) + alpha * (w_23[count]-w_23[count-1])
                w_23.append(w_23_new)

            count = count + 1
        correct = 0
        for i in range(150-int(percentage*150)):
            x_1 = training_set[i:i+1,0:5]
            v_2 = np.dot(x_1,w_12[count])
            v_2 = np.concatenate((np.ones((1,1)),v_2),axis=1)
            y_2 = 1/(1+np.exp((-1)*v_2))
            v_3 = np.dot(y_2,w_23[count])
            y_3 = 1/(1+np.exp((-1)*v_3))
            max = np.amax(y_3) 
            if (y_3[0,0] == max) and (training_set[i,5]==1):
                correct=correct+1
            if (y_3[0,1] == max) and (training_set[i,5]==2):
                correct=correct+1
            if (y_3[0,2] == max) and (training_set[i,5]==3):
                correct=correct+1
        accuracy = float(correct)/float(150-percentage*150)
        error.append(1-accuracy)
    return error
'''
# TEST MODEL:
correct = 0
for i in range(150-int(percentage*150)):
    x_1 = training_set[i:i+1,0:5]
    v_2 = np.dot(x_1,w_12[count])
    v_2 = np.concatenate((np.ones((1,1)),v_2),axis=1)
    y_2 = 1/(1+np.exp((-1)*v_2))
    v_3 = np.dot(y_2,w_23[count])
    y_3 = 1/(1+np.exp((-1)*v_3))
    max = np.amax(y_3) 
    if (y_3[0,0] == max) and (training_set[i,5]==1):
        correct=correct+1
    if (y_3[0,1] == max) and (training_set[i,5]==2):
        correct=correct+1
    if (y_3[0,2] == max) and (training_set[i,5]==3):
        correct=correct+1
accuracy = float(correct)/float(150-percentage*150)


print("Accuracy: ")
print(accuracy)
'''
x= []
error1 = train_model(1,0.1)
error2 = train_model(0.1,0.1)
error3 = train_model(5,0.1)
error4 = train_model(10,0.1)
error10 = []
error20 = []
error30 = []
error40 = []

for i in range(100):
    if i%5==0:
        x.append(i)
        error10.append(error1[i])
        error20.append(error2[i])
        error30.append(error3[i])
        error40.append(error4[i])



plt.plot(x,error10,label = "lr = 1")
plt.plot(x,error20,label = "lr = 0.1")
plt.plot(x,error30,label = "lr = 5")
plt.plot(x,error40,label = "lr = 10")

plt.xlabel("Epoch")
plt.ylabel("Error rate ")
plt.title("Error trajectory Multi Layered Neural Network")
plt.legend()
plt.show()       









    
    
