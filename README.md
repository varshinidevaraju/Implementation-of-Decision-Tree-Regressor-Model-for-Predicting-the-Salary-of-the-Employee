# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.
 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: D.VARSHINI
RegisterNumber:  212223230234
*/

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
## Data Head:
![image](https://github.com/user-attachments/assets/8e4aa222-11fd-44a5-951a-de73dfb11e75)


## Data Info:
![image](https://github.com/user-attachments/assets/f00f97b5-3380-46d5-886b-faf5ef40393a)


## isnull().sum():
![image](https://github.com/user-attachments/assets/d90169ac-0a86-4b64-aa60-90b55091a602)


## Data Head for salary:
![image](https://github.com/user-attachments/assets/ad16aeee-62c8-4811-afcf-e0a2526e7d98)

## Mean Squared Error :
![image](https://github.com/user-attachments/assets/040cae1c-c02a-44f5-bd3e-9a9ea0ff248a)


## r2 Value:
![image](https://github.com/user-attachments/assets/2f79d241-6f92-4dff-838c-66c6e2e26b65)

## Data prediction :
![image](https://github.com/user-attachments/assets/fcae0751-1864-4f1e-a093-b49c5b297952)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
