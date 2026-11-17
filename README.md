# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Python libraries and load the Employee.csv dataset using Pandas.
  
2. Select the input features (X) and target variable (Y) from the dataset.
 
3. Split the dataset into 80% training data and 20% testing data.

4. Train the Decision Tree Classifier using the Gini criterion and predict employee churn.
  
5. Evaluate the model using accuracy, confusion matrix and classification report, and predict whether a new employee will stay or leave.
## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Shanmuga priya K
RegisterNumber: 212225040401
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data = pd.read_csv("Employee.csv")

print("First five records of the dataset:")
print(data.head())

X = data[
    [
        "satisfaction_level",
        "last_evaluation",
        "number_project",
        "average_montly_hours",
        "time_spend_company",
        "Work_accident",
        "promotion_last_5years"
    ]
]

y = data["left"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)


model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nActual Churn Status:")
print(y_test.values[:20])

print("\nPredicted Churn Status:")
print(y_pred[:20])
print("\nAccuracy:", accuracy)

print("\nAccuracy Percentage:", accuracy * 100, "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_employee = pd.DataFrame({
    "satisfaction_level": [0.35],
    "last_evaluation": [0.80],
    "number_project": [5],
    "average_montly_hours": [250],
    "time_spend_company": [5],
    "Work_accident": [0],
    "promotion_last_5years": [0]
})


prediction = model.predict(new_employee)

print("\nNew Employee Details:")
print("Satisfaction Level: 0.35")
print("Last Evaluation: 0.80")
print("Number of Projects: 5")
print("Average Monthly Hours: 250")
print("Time Spent in Company: 5 years")
print("Work Accident: 0")
print("Promotion in Last 5 Years: 0")

if prediction[0] == 1:
    print("\nPredicted Churn Status: Employee will leave")
else:
    print("\nPredicted Churn Status: Employee will stay")
```

## Output:

<img width="1713" height="525" alt="image" src="https://github.com/user-attachments/assets/9dd85064-b9b1-4609-9cef-46b07445a293" />

<img width="1752" height="488" alt="image" src="https://github.com/user-attachments/assets/6aab8e92-4581-4219-9a42-5faa92c23d84" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
