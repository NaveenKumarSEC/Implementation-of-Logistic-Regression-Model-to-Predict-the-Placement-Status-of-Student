# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
   Display the results.


## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Placement_Data.csv")   

print("Dataset Preview:")
print(data.head())

data = data.drop(["sl_no", "salary"], axis=1)

data["status"] = data["status"].map({"Placed": 1, "Not Placed": 0})

X = data.drop("status", axis=1)
y = data["status"]

X = pd.get_dummies(X, drop_first=True)

print("\nAfter Encoding:")
print(X.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Placement Prediction")
plt.show()
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Naveenkumar M
RegisterNumber:  212224230182
*/
```

## Output:
<img width="732" height="831" alt="image" src="https://github.com/user-attachments/assets/e3baf7c0-51d5-4d13-bea6-156fff0ab02e" />

<img width="707" height="798" alt="image" src="https://github.com/user-attachments/assets/51228eb6-dcaa-4a40-be64-a463066accc5" />




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
