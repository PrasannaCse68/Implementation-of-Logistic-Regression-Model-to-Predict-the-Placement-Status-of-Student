# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
NAME: PRASANNA GR
REG NO:212221040129
```
```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
![image](https://github.com/PrasannaCse68/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/127935950/2e729154-ff21-406e-b7af-615c0add1099)
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]
```

## Output:

Original data(first five columns:

![image](https://github.com/PrasannaCse68/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/127935950/ba4ce184-c51e-431a-9ea9-d0d7fb29ce75)



Data after dropping unwanted columns(first five):
![image](https://github.com/PrasannaCse68/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/127935950/78a0dcd3-b4d0-4d8a-9e7e-3ae1964b6630)


Checking the presence of null values:
![image](https://github.com/PrasannaCse68/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/127935950/92302f12-84d5-4c30-b38c-898c039c9d3e)


Checking the presence of duplicated values
![image](https://github.com/PrasannaCse68/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/127935950/b2b904ba-df8d-4465-a0fa-508b1e4fe110)



Data after Encoding
![image](https://github.com/PrasannaCse68/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/127935950/6962ef2b-f41a-4224-9d08-b11c795959d1)


X DATA:
![image](https://github.com/PrasannaCse68/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/127935950/02c8cc57-1f41-4304-80b0-30f71757704c)



Y DATA:
![image](https://github.com/PrasannaCse68/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/127935950/845e6ca6-383a-4531-bd71-7d1cf1f49e10)


Predicted Values:
![image](https://github.com/PrasannaCse68/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/127935950/fa9056d7-b036-4a80-a74f-7530f02d428d)



Accuracy Score:
![image](https://github.com/PrasannaCse68/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/127935950/3f9e3653-4984-41f4-b012-ea39539cd812)



Confusion Matrix:
![image](https://github.com/PrasannaCse68/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/127935950/0b63e562-9f23-4961-8b9d-ffefe1f0e150)



Classification Report:

![image](https://github.com/PrasannaCse68/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/127935950/65a46900-c2e3-4e03-9a9f-893c7254fd66)



Predicting output from Regression Model:
![image](https://github.com/PrasannaCse68/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/127935950/fc037c5f-584f-470a-82db-92e8eedf7a89)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
