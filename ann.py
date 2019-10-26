#PART1- DATA PREPROCESSING...
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset= pd.read_csv('Churn_Modelling.csv')
print(dataset)
x = dataset.iloc[:,3:13].values #as we are selectng column so we will choose after the comma.....x contains the independent varriable
print(x)
y=dataset.iloc[:,13].values #here y contains the label value
print(y)
#here we will encode our categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#here we encode all the text value to numerical value to encode it we have used LabelEncoder
labelencoder_x1=LabelEncoder()
x[:,1]=labelencoder_x1.fit_transform(x[:,1])
#print(x[:,1])
labelencoder_x2=LabelEncoder()
x[:,2]=labelencoder_x2.fit_transform(x[:,2])
#print(x[:,2])
#now we have to create catogerical varriable i.e. dummy varriable  which will only reside 2 classes.....
dummy=OneHotEncoder(categorical_features=[1])
x=dummy.fit_transform(x).toarray() # this will change the object value of the x to float64 and make it divided into 2 classes
x= x[:, 1:]

#spliting the independent and dependent varriable...
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0) #to train the ann on 8000 observation and test it on 2000 observation

#feature scaling  using Standardscaler  
#Standardscaler   will transform our data the formula is (xi-xmean)/standard deviation of that feature
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()#we have to create its object to call fit_transform
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
#PART2:-CREATING  ANN model...
import keras  #keras will build the neural network....
from keras.models import Sequential #to initialize our NN
from keras.layers import Dense #it will used to create layers of our ANN
'''initializing of ANN  we have 2 ways to initialize our ANN
1.defining sequence of layers
2. defining a graph'''
#here we will use the 1st method and as we have  a classification problem so this neural network is going to be classifier ....

classifier = Sequential() #making object of sequential class

#adding  input layer and the first hidden layer
classifier.add(Dense(output_dim = 6 ,init= 'uniform',activation= 'relu' , input_dim=11))
 #adding hidden layer
classifier.add(Dense(output_dim=6 ,init= 'uniform',activation='relu'))

#adding output layer
classifier.add(Dense(output_dim=1 ,init= 'uniform',activation='sigmoid')) #as here we have only 1 catogerical varriable so can use sigmoid function but if there are more than 1 catogerical varriable then we have to use softmax function

#compiling the ANN
classifier.compile(optimizer='adam' , loss= 'binary_crossentropy' , metrics=['accuracy'])  #here we are using   adam algorithm to have the best weight for our neural network and as here have binary output i.e. one output so we are using binary_crossentropy

#fitting the ANN to the Training set
classifier.fit(x_train,y_train, batch_size=10, nb_epoch=100)

y_pred = classifier.predict(x_test)

y_pred = (y_pred>0.5)
#Creation of confusion matrix to know the accuracy..... (number of correct prediction)/total number of predication

from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_test,y_pred)
