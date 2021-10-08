import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


df = pd.read_csv("/Users/apple/Desktop/Coding/Sem5/Assignment2_Code/iris.data")


data_setosa = df.iloc[0:50,0:4].values
data_versicolor = df.iloc[50:100,0:4].values
data_virginica = df.iloc[100:150,0:4].values

# Using Backpropogation Algorithm to classify 
# Setosa is of class 1, versicolor is of class 2, virginica is of class 3
data_setosa = data_setosa.astype('float64')
data_versicolor = data_versicolor.astype('float64')
data_virginica = data_virginica.astype('float64')
data_setosa = (0.001)*data_setosa
data_versicolor = (0.001)*data_versicolor
data_virginica = (0.001)*data_virginica
a=np.ones((50,1))
a=a.astype('float64')
data_class_setosa = np.concatenate((0.001*a,data_setosa,np.ones((50,1))),axis = 1)
data_class_versicolor = np.concatenate((0.001*a,data_versicolor,np.ones((50,1))*2),axis = 1)
data_class_virginica = np.concatenate((0.001*a,data_virginica,np.ones((50,1))*3),axis = 1)
data_class = np.concatenate((data_class_setosa,data_class_versicolor,data_class_virginica),axis = 0)
np.random.shuffle(data_class)
percentage = 0.8                # Percentage of training set
training_set = data_class[0:int(percentage*150),:]       # Training data
test_set = data_class[int(percentage*150):150,:]         # Test data

# Train weight vector
lr = 0.5                                                   # learning rate
w1_1 = np.random.rand(1,5)                                   # weight vector for P(y=1) (1x5)
w2_1 = np.random.rand(1,5)                                   # weight vector for P(y=2) (1x5)
w3_1 = np.random.rand(1,5)                                   # weight vector for P(y=3) (1x5)
error1 =[]
error2 =[]
error3 =[]
error4 =[]
error5 =[]
for j in range(100):
    
    np.random.shuffle(training_set)
    sum1 = np.zeros((1,5))                                    # gradient vector for w1 (1x5)
    sum2 = np.zeros((1,5))                                    # gradient vector for w2 (1x5)
    sum3 = np.zeros((1,5))                                    # gradient vector for w3 (1x5)
    for i in range(int(percentage*150)):
        temp_1 = (np.dot(w1_1,np.transpose(training_set[i:i+1,0:5])))       # w1T*xi
        temp1 = np.exp(temp_1[0,0])                                    # exp(w1T*xi)
        temp_2 = (np.dot(w2_1,np.transpose(training_set[i:i+1,0:5])))       # w2T*xi
        temp2 = np.exp(temp_2[0,0])                                    # exp(w2T*xi) 
        temp_3 = (np.dot(w3_1,np.transpose(training_set[i:i+1,0:5])))       # w3T*xi 
        temp3 = np.exp(temp_3[0,0])                                    # exp(w3T*xi) 
        temp4 = temp1 + temp2 + temp3
        #print(temp1/temp3,temp2/temp3)
        if training_set[i,5] == 1:
            sum1 = sum1 - (training_set[i:i+1,0:5])
        if training_set[i,5] == 2:
            sum2 = sum2 - (training_set[i:i+1,0:5])
        if training_set[i,5] == 3:
            sum3 = sum3 - (training_set[i:i+1,0:5])
        sum1 = sum1 + ((training_set[i:i+1,0:5])/(1+np.exp(temp_2-temp_1)+np.exp(temp_3-temp_1)))
        sum2 = sum2 + ((training_set[i:i+1,0:5])/(1+np.exp(temp_1-temp_2)+np.exp(temp_3-temp_2)))
        sum3 = sum3 + ((training_set[i:i+1,0:5])/(1+np.exp(temp_1-temp_3)+np.exp(temp_2-temp_3)))
        
     # Calculate Error:
    correct = 0
    for i in range(int(150-percentage*150)):
        temp_1 = (np.dot(w1_1,np.transpose(training_set[i:i+1,0:5])))       # w1T*xi
        temp1 = np.exp(temp_1[0,0])                                    # exp(w1T*xi)
        temp_2 = (np.dot(w2_1,np.transpose(training_set[i:i+1,0:5])))       # w2T*xi
        temp2 = np.exp(temp_2[0,0])                                    # exp(w2T*xi) 
        temp_3 = (np.dot(w3_1,np.transpose(training_set[i:i+1,0:5])))       # w3T*xi 
        temp3 = np.exp(temp_3[0,0])                                    # exp(w3T*xi) 
        temp4 = temp1 + temp2 + temp3
        res_1 = 1/(1+np.exp(temp_2-temp_1)+np.exp(temp_3-temp_1))
        res_2 = 1/(1+np.exp(temp_1-temp_2)+np.exp(temp_3-temp_2))
        res_3 = 1/(1+np.exp(temp_1-temp_3)+np.exp(temp_2-temp_3))
        #print(res_1,res_2,res_3)
        if res_1 >= res_2 and res_1 >= res_3 and test_set[i,5] == 1:
            correct = correct + 1 
        if res_2 >= res_3 and res_2 >= res_3 and test_set[i,5] == 2:
            correct = correct + 1
        if res_3 >= res_2 and res_3 >= res_1 and test_set[i,5] == 3:
            correct = correct + 1
    accuracy = float(correct)/float(150-percentage*150)  
    error1.append(1-accuracy)   
    
    w1_1 = w1_1 + lr*sum1
    w2_1 = w2_1 + lr*sum2
    w3_1 = w3_1 + lr*sum3
w1_2 = np.random.rand(1,5)                                   # weight vector for P(y=1) (1x5)
w2_2 = np.random.rand(1,5)                                   # weight vector for P(y=2) (1x5)
w3_2 = np.random.rand(1,5)                                   # weight vector for P(y=3) (1x5)
for j in range(100):
    
    np.random.shuffle(training_set)
    sum1 = np.zeros((1,5))                                    # gradient vector for w1 (1x5)
    sum2 = np.zeros((1,5))                                    # gradient vector for w2 (1x5)
    sum3 = np.zeros((1,5))                                    # gradient vector for w3 (1x5)
    for i in range(int(percentage*150)):
        temp_1 = (np.dot(w1_2,np.transpose(training_set[i:i+1,0:5])))       # w1T*xi
        temp1 = np.exp(temp_1[0,0])                                    # exp(w1T*xi)
        temp_2 = (np.dot(w2_2,np.transpose(training_set[i:i+1,0:5])))       # w2T*xi
        temp2 = np.exp(temp_2[0,0])                                    # exp(w2T*xi) 
        temp_3 = (np.dot(w3_2,np.transpose(training_set[i:i+1,0:5])))       # w3T*xi 
        temp3 = np.exp(temp_3[0,0])                                    # exp(w3T*xi) 
        temp4 = temp1 + temp2 + temp3
        #print(temp1/temp3,temp2/temp3)
        if training_set[i,5] == 1:
            sum1 = sum1 - (training_set[i:i+1,0:5])
        if training_set[i,5] == 2:
            sum2 = sum2 - (training_set[i:i+1,0:5])
        if training_set[i,5] == 3:
            sum3 = sum3 - (training_set[i:i+1,0:5])
        sum1 = sum1 + ((training_set[i:i+1,0:5])/(1+np.exp(temp_2-temp_1)+np.exp(temp_3-temp_1)))
        sum2 = sum2 + ((training_set[i:i+1,0:5])/(1+np.exp(temp_1-temp_2)+np.exp(temp_3-temp_2)))
        sum3 = sum3 + ((training_set[i:i+1,0:5])/(1+np.exp(temp_1-temp_3)+np.exp(temp_2-temp_3)))
        
     # Calculate Error:
    correct = 0
    for i in range(int(150-percentage*150)):
        temp_1 = (np.dot(w1_2,np.transpose(training_set[i:i+1,0:5])))       # w1T*xi
        temp1 = np.exp(temp_1[0,0])                                    # exp(w1T*xi)
        temp_2 = (np.dot(w2_2,np.transpose(training_set[i:i+1,0:5])))       # w2T*xi
        temp2 = np.exp(temp_2[0,0])                                    # exp(w2T*xi) 
        temp_3 = (np.dot(w3_2,np.transpose(training_set[i:i+1,0:5])))       # w3T*xi 
        temp3 = np.exp(temp_3[0,0])                                    # exp(w3T*xi) 
        temp4 = temp1 + temp2 + temp3
        res_1 = 1/(1+np.exp(temp_2-temp_1)+np.exp(temp_3-temp_1))
        res_2 = 1/(1+np.exp(temp_1-temp_2)+np.exp(temp_3-temp_2))
        res_3 = 1/(1+np.exp(temp_1-temp_3)+np.exp(temp_2-temp_3))
        #print(res_1,res_2,res_3)
        if res_1 >= res_2 and res_1 >= res_3 and test_set[i,5] == 1:
            correct = correct + 1 
        if res_2 >= res_3 and res_2 >= res_3 and test_set[i,5] == 2:
            correct = correct + 1
        if res_3 >= res_2 and res_3 >= res_1 and test_set[i,5] == 3:
            correct = correct + 1
    accuracy = float(correct)/float(150-percentage*150)  
    error2.append(1-accuracy)   
    
    w1_2 = w1_2 + lr*sum1
    w2_2 = w2_2 + lr*sum2
    w3_2 = w3_2 + lr*sum3
w1_3 = np.random.rand(1,5)                                   # weight vector for P(y=1) (1x5)
w2_3 = np.random.rand(1,5)                                   # weight vector for P(y=2) (1x5)
w3_3 = np.random.rand(1,5)                                   # weight vector for P(y=3) (1x5)
for j in range(100):
    
    np.random.shuffle(training_set)
    sum1 = np.zeros((1,5))                                    # gradient vector for w1 (1x5)
    sum2 = np.zeros((1,5))                                    # gradient vector for w2 (1x5)
    sum3 = np.zeros((1,5))                                    # gradient vector for w3 (1x5)
    for i in range(int(percentage*150)):
        temp_1 = (np.dot(w1_3,np.transpose(training_set[i:i+1,0:5])))       # w1T*xi
        temp1 = np.exp(temp_1[0,0])                                    # exp(w1T*xi)
        temp_2 = (np.dot(w2_3,np.transpose(training_set[i:i+1,0:5])))       # w2T*xi
        temp2 = np.exp(temp_2[0,0])                                    # exp(w2T*xi) 
        temp_3 = (np.dot(w3_3,np.transpose(training_set[i:i+1,0:5])))       # w3T*xi 
        temp3 = np.exp(temp_3[0,0])                                    # exp(w3T*xi) 
        temp4 = temp1 + temp2 + temp3
        #print(temp1/temp3,temp2/temp3)
        if training_set[i,5] == 1:
            sum1 = sum1 - (training_set[i:i+1,0:5])
        if training_set[i,5] == 2:
            sum2 = sum2 - (training_set[i:i+1,0:5])
        if training_set[i,5] == 3:
            sum3 = sum3 - (training_set[i:i+1,0:5])
        sum1 = sum1 + ((training_set[i:i+1,0:5])/(1+np.exp(temp_2-temp_1)+np.exp(temp_3-temp_1)))
        sum2 = sum2 + ((training_set[i:i+1,0:5])/(1+np.exp(temp_1-temp_2)+np.exp(temp_3-temp_2)))
        sum3 = sum3 + ((training_set[i:i+1,0:5])/(1+np.exp(temp_1-temp_3)+np.exp(temp_2-temp_3)))
        
     # Calculate Error:
    correct = 0
    for i in range(int(150-percentage*150)):
        temp_1 = (np.dot(w1_3,np.transpose(training_set[i:i+1,0:5])))       # w1T*xi
        temp1 = np.exp(temp_1[0,0])                                    # exp(w1T*xi)
        temp_2 = (np.dot(w2_3,np.transpose(training_set[i:i+1,0:5])))       # w2T*xi
        temp2 = np.exp(temp_2[0,0])                                    # exp(w2T*xi) 
        temp_3 = (np.dot(w3_3,np.transpose(training_set[i:i+1,0:5])))       # w3T*xi 
        temp3 = np.exp(temp_3[0,0])                                    # exp(w3T*xi) 
        temp4 = temp1 + temp2 + temp3
        res_1 = 1/(1+np.exp(temp_2-temp_1)+np.exp(temp_3-temp_1))
        res_2 = 1/(1+np.exp(temp_1-temp_2)+np.exp(temp_3-temp_2))
        res_3 = 1/(1+np.exp(temp_1-temp_3)+np.exp(temp_2-temp_3))
        #print(res_1,res_2,res_3)
        if res_1 >= res_2 and res_1 >= res_3 and test_set[i,5] == 1:
            correct = correct + 1 
        if res_2 >= res_3 and res_2 >= res_3 and test_set[i,5] == 2:
            correct = correct + 1
        if res_3 >= res_2 and res_3 >= res_1 and test_set[i,5] == 3:
            correct = correct + 1
    accuracy = float(correct)/float(150-percentage*150)  
    error3.append(1-accuracy)   
    
    w1_3 = w1_3 + lr*sum1
    w2_3 = w2_3 + lr*sum2
    w3_3 = w3_3 + lr*sum3
w1_4 = np.random.rand(1,5)                                   # weight vector for P(y=1) (1x5)
w2_4 = np.random.rand(1,5)                                   # weight vector for P(y=2) (1x5)
w3_4 = np.random.rand(1,5)                                   # weight vector for P(y=3) (1x5)
for j in range(100):
    
    np.random.shuffle(training_set)
    sum1 = np.zeros((1,5))                                    # gradient vector for w1 (1x5)
    sum2 = np.zeros((1,5))                                    # gradient vector for w2 (1x5)
    sum3 = np.zeros((1,5))                                    # gradient vector for w3 (1x5)
    for i in range(int(percentage*150)):
        temp_1 = (np.dot(w1_4,np.transpose(training_set[i:i+1,0:5])))       # w1T*xi
        temp1 = np.exp(temp_1[0,0])                                    # exp(w1T*xi)
        temp_2 = (np.dot(w2_4,np.transpose(training_set[i:i+1,0:5])))       # w2T*xi
        temp2 = np.exp(temp_2[0,0])                                    # exp(w2T*xi) 
        temp_3 = (np.dot(w3_4,np.transpose(training_set[i:i+1,0:5])))       # w3T*xi 
        temp3 = np.exp(temp_3[0,0])                                    # exp(w3T*xi) 
        temp4 = temp1 + temp2 + temp3
        #print(temp1/temp3,temp2/temp3)
        if training_set[i,5] == 1:
            sum1 = sum1 - (training_set[i:i+1,0:5])
        if training_set[i,5] == 2:
            sum2 = sum2 - (training_set[i:i+1,0:5])
        if training_set[i,5] == 3:
            sum3 = sum3 - (training_set[i:i+1,0:5])
        sum1 = sum1 + ((training_set[i:i+1,0:5])/(1+np.exp(temp_2-temp_1)+np.exp(temp_3-temp_1)))
        sum2 = sum2 + ((training_set[i:i+1,0:5])/(1+np.exp(temp_1-temp_2)+np.exp(temp_3-temp_2)))
        sum3 = sum3 + ((training_set[i:i+1,0:5])/(1+np.exp(temp_1-temp_3)+np.exp(temp_2-temp_3)))
        
     # Calculate Error:
    correct = 0
    for i in range(int(150-percentage*150)):
        temp_1 = (np.dot(w1_4,np.transpose(training_set[i:i+1,0:5])))       # w1T*xi
        temp1 = np.exp(temp_1[0,0])                                    # exp(w1T*xi)
        temp_2 = (np.dot(w2_4,np.transpose(training_set[i:i+1,0:5])))       # w2T*xi
        temp2 = np.exp(temp_2[0,0])                                    # exp(w2T*xi) 
        temp_3 = (np.dot(w3_4,np.transpose(training_set[i:i+1,0:5])))       # w3T*xi 
        temp3 = np.exp(temp_3[0,0])                                    # exp(w3T*xi) 
        temp4 = temp1 + temp2 + temp3
        res_1 = 1/(1+np.exp(temp_2-temp_1)+np.exp(temp_3-temp_1))
        res_2 = 1/(1+np.exp(temp_1-temp_2)+np.exp(temp_3-temp_2))
        res_3 = 1/(1+np.exp(temp_1-temp_3)+np.exp(temp_2-temp_3))
        #print(res_1,res_2,res_3)
        if res_1 >= res_2 and res_1 >= res_3 and test_set[i,5] == 1:
            correct = correct + 1 
        if res_2 >= res_3 and res_2 >= res_3 and test_set[i,5] == 2:
            correct = correct + 1
        if res_3 >= res_2 and res_3 >= res_1 and test_set[i,5] == 3:
            correct = correct + 1
    accuracy = float(correct)/float(150-percentage*150)  
    error4.append(1-accuracy)   
    
    w1_4 = w1_4 + lr*sum1
    w2_4 = w2_4 + lr*sum2
    w3_4 = w3_4 + lr*sum3
w1_5 = np.random.rand(1,5)                                   # weight vector for P(y=1) (1x5)
w2_5 = np.random.rand(1,5)                                   # weight vector for P(y=2) (1x5)
w3_5 = np.random.rand(1,5)                                   # weight vector for P(y=3) (1x5)
for j in range(100):
    
    np.random.shuffle(training_set)
    sum1 = np.zeros((1,5))                                    # gradient vector for w1 (1x5)
    sum2 = np.zeros((1,5))                                    # gradient vector for w2 (1x5)
    sum3 = np.zeros((1,5))                                    # gradient vector for w3 (1x5)
    for i in range(int(percentage*150)):
        temp_1 = (np.dot(w1_5,np.transpose(training_set[i:i+1,0:5])))       # w1T*xi
        temp1 = np.exp(temp_1[0,0])                                    # exp(w1T*xi)
        temp_2 = (np.dot(w2_5,np.transpose(training_set[i:i+1,0:5])))       # w2T*xi
        temp2 = np.exp(temp_2[0,0])                                    # exp(w2T*xi) 
        temp_3 = (np.dot(w3_5,np.transpose(training_set[i:i+1,0:5])))       # w3T*xi 
        temp3 = np.exp(temp_3[0,0])                                    # exp(w3T*xi) 
        temp4 = temp1 + temp2 + temp3
        #print(temp1/temp3,temp2/temp3)
        if training_set[i,5] == 1:
            sum1 = sum1 - (training_set[i:i+1,0:5])
        if training_set[i,5] == 2:
            sum2 = sum2 - (training_set[i:i+1,0:5])
        if training_set[i,5] == 3:
            sum3 = sum3 - (training_set[i:i+1,0:5])
        sum1 = sum1 + ((training_set[i:i+1,0:5])/(1+np.exp(temp_2-temp_1)+np.exp(temp_3-temp_1)))
        sum2 = sum2 + ((training_set[i:i+1,0:5])/(1+np.exp(temp_1-temp_2)+np.exp(temp_3-temp_2)))
        sum3 = sum3 + ((training_set[i:i+1,0:5])/(1+np.exp(temp_1-temp_3)+np.exp(temp_2-temp_3)))
        
     # Calculate Error:
    correct = 0
    for i in range(int(150-percentage*150)):
        temp_1 = (np.dot(w1_5,np.transpose(training_set[i:i+1,0:5])))       # w1T*xi
        temp1 = np.exp(temp_1[0,0])                                    # exp(w1T*xi)
        temp_2 = (np.dot(w2_5,np.transpose(training_set[i:i+1,0:5])))       # w2T*xi
        temp2 = np.exp(temp_2[0,0])                                    # exp(w2T*xi) 
        temp_3 = (np.dot(w3_5,np.transpose(training_set[i:i+1,0:5])))       # w3T*xi 
        temp3 = np.exp(temp_3[0,0])                                    # exp(w3T*xi) 
        temp4 = temp1 + temp2 + temp3
        res_1 = 1/(1+np.exp(temp_2-temp_1)+np.exp(temp_3-temp_1))
        res_2 = 1/(1+np.exp(temp_1-temp_2)+np.exp(temp_3-temp_2))
        res_3 = 1/(1+np.exp(temp_1-temp_3)+np.exp(temp_2-temp_3))
        #print(res_1,res_2,res_3)
        if res_1 >= res_2 and res_1 >= res_3 and test_set[i,5] == 1:
            correct = correct + 1 
        if res_2 >= res_3 and res_2 >= res_3 and test_set[i,5] == 2:
            correct = correct + 1
        if res_3 >= res_2 and res_3 >= res_1 and test_set[i,5] == 3:
            correct = correct + 1
    accuracy = float(correct)/float(150-percentage*150)  
    error5.append(1-accuracy)   
    
    w1_5 = w1_5 + lr*sum1
    w2_5 = w2_5 + lr*sum2
    w3_5 = w3_5 + lr*sum3

'''
# Test weight vector
correct = 0
for i in range(int(150-percentage*150)):
    temp_1 = (np.dot(w1,np.transpose(training_set[i:i+1,0:5])))       # w1T*xi
    temp1 = np.exp(temp_1[0,0])                                    # exp(w1T*xi)
    temp_2 = (np.dot(w2,np.transpose(training_set[i:i+1,0:5])))       # w2T*xi
    temp2 = np.exp(temp_2[0,0])                                    # exp(w2T*xi) 
    temp_3 = (np.dot(w3,np.transpose(training_set[i:i+1,0:5])))       # w3T*xi 
    temp3 = np.exp(temp_3[0,0])                                    # exp(w3T*xi) 
    temp4 = temp1 + temp2 + temp3
    res_1 = 1/(1+np.exp(temp_2-temp_1)+np.exp(temp_3-temp_1))
    res_2 = 1/(1+np.exp(temp_1-temp_2)+np.exp(temp_3-temp_2))
    res_3 = 1/(1+np.exp(temp_1-temp_3)+np.exp(temp_2-temp_3))
    #print(res_1,res_2,res_3)
    if res_1 >= res_2 and res_1 >= res_3 and test_set[i,5] == 1:
        correct = correct + 1 
    if res_2 >= res_3 and res_2 >= res_3 and test_set[i,5] == 2:
        correct = correct + 1
    if res_3 >= res_2 and res_3 >= res_1 and test_set[i,5] == 3:
        correct = correct + 1
accuracy = float(correct)/float(150-percentage*150)
print("Accuracy: ")
print(accuracy)
'''
x=[]
for i in range(100):
    x.append(i)
plt.plot(x,error1,label = "sample 1")
plt.plot(x,error2,label = "sample 2")
plt.plot(x,error3,label = "sample 3")
plt.plot(x,error4,label = "sample 4")
plt.plot(x,error5,label = "sample 5")
plt.xlabel("Epoch")
plt.ylabel("Error rate ")
plt.title("Error trajectory for Logistic Regression ")
plt.legend()
plt.show()

