# importing libraries

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle
#reading csv file and Feature selection

data = pd.read_csv('student-mat.csv',sep=";")
features = ["G1", "G2", "G3", "studytime", "failures", "absences"]

data = data[features]

#selection of target column and Data cleaning

target = "G3"
X = np.array(data.drop([target],1))
y = np.array(data[target])

#splitting the data into training and testing data

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.1)

#training the LinearRegression model multiple times to reach good accuracy

"""
best=0
for i in range(30):
    x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train,y_train)

#computing accuracy of the model 

    acc = linear.score(x_test,y_test)
    print(acc)

    if acc > best:
        best=acc
        
#storing the most accurate model in a pickle file        

        with open("studentmodel.pickle","wb") as f:
            pickle.dump(linear,f)"""

#using the best model stored in pickle file to make predictions

pickle_in = open("studentmodel.pickle","rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print('|--Predicted Grades--|--Features--|--Actual Grades--|')

#doing predictions on test data

predictions = linear.predict(x_test)

for i in range(len(predictions)):
    print(int(predictions[i]),x_test[i],y_test[i],sep=" | ")

a=[]
for i in range(len(x_test)):
    a.append(x_test[i][3])







