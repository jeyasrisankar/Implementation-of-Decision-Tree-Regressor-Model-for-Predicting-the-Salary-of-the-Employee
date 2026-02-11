# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset containing employee level and salary values.

2. Select features and target, where Level is the independent variable and Salary is the dependent variable.

3. Split the dataset into training and testing sets.

4. Create a Decision Tree Regressor by selecting the splitting criterion and maximum depth.

5. Train the model using the training data.

6. Predict salary values for test data or new employee levels and evaluate the performance.
 
 
 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: JEYASRI S
RegisterNumber:  212225040155
*/

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\acer\Downloads\salary.csv")

print("Dataset Preview:")
print(df.head())

X = df[["Level"]]   
y = df["Salary"]           

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("MAE  :", mean_absolute_error(y_test, y_pred))
print("MSE  :", mse)
print("RMSE :", rmse)
print("R2   :", r2_score(y_test, y_pred))

plt.figure(figsize=(16, 10))
plot_tree(
    model,
    feature_names=["Level"],
    filled=True
)
plt.title("Decision Tree Regressor for Employee Salary Prediction")
plt.show()

new_exp = [[5]]  
predicted_salary = model.predict(new_exp)
print("\nPredicted Salary for 5 years experience:", predicted_salary[0])

```

## Output:



<img width="747" height="292" alt="Screenshot 2026-02-11 113303" src="https://github.com/user-attachments/assets/b0ea93b3-e50c-4ec1-a3e5-6afce2a0858a" />



<img width="1682" height="755" alt="Screenshot 2026-02-11 113332" src="https://github.com/user-attachments/assets/d10dc63d-3cf6-42d0-b93e-2bff7ec86739" />




<img width="597" height="106" alt="Screenshot 2026-02-11 113342" src="https://github.com/user-attachments/assets/66b3370f-9587-453c-8c99-969421c1be5f" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
