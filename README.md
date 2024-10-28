# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Start the Program.

Step 2. Import the necessary packages.

Step 3. Read the given csv file and display the few contents of the data.

Step 4. Assign the features for x and y respectively.

Step 5. Split the x and y sets into train and test sets.

Step 6. Convert the Alphabetical data to numeric using CountVectorizer.

Step 7. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

Step 8. Find the accuracy of the model.

Step 9. End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: KAVIYA SNEKA M
RegisterNumber:  212223040091
*/
```
```
import pandas as pd
data=pd.read_csv('spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix, classification_report
con = confusion_matrix(y_test,y_pred)
con
cl=classification_report(y_test,y_pred)
```

## Output:
##Head:
![WhatsApp Image 2024-10-28 at 10 11 54_fd66586d](https://github.com/user-attachments/assets/ff72590b-3f13-4f3d-83db-7ffe27501774)

##Info:
![WhatsApp Image 2024-10-28 at 10 12 06_12ce6cbf](https://github.com/user-attachments/assets/f07846ea-4a47-4a78-af0d-a131e8720e66)

##Acuuracy:
![WhatsApp Image 2024-10-28 at 10 12 11_f1465362](https://github.com/user-attachments/assets/e6bd179a-436d-4398-90dd-bb30316b8cd6)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
