# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries in python required for finding Gradient Design.
2. Read the dataset file and check any null value using .isnull() method.
3. Declare the default variables with respective values for linear regression.
4. Calculate the loss using Mean Square Error.
5. Predict the value of y.
6. Plot the graph respect to hours and scores using .scatterplot() method for Linear Regression.
7. Plot the graph respect to loss and iterations using .plot() method for Gradient Descent.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Naveen Kumar E
RegisterNumber: 212222220029
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X for the intercept term
    X = np.c_[np.ones(len(X)), X]

    # Initialize theta with zeros
    theta = np.zeros(X.shape[1]).reshape(-1, 1)

    # Perform gradient descent
    for _ in range(num_iters):
        # Calculate predictions
        predictions = X.dot(theta).reshape(-1, 1)

        # Calculate errors
        errors = (predictions - y).reshape(-1, 1)

        # Update theta using gradient descent
        theta -= learning_rate * (2 / len(X)) * X.T.dot(errors)

    return theta

# Read data from CSV file
data = pd.read_csv('/content/50_Startups.csv', header=None)
data.head()
X = data.iloc[1:, :-2].values.astype(float)
y = data.iloc[1:, -1].values.reshape(-1, 1)

# Standardize features and target variable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

# Example usage
# X_array = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
# y_array = np.array([2, 7, 11, 16])

# Learn model parameters
theta_result = linear_regression(X_scaled, y_scaled)

# Predict target value for a new data point
new_data = np.array([165349.2, 136897.8, 471784.1]).reshape(-1, 1)
new_scaled = scaler.fit_transform(new_data)

prediction = np.dot(np.append(1, new_scaled), theta_result)
prediction = prediction.reshape(-1, 1)

# Inverse transform the prediction to get the original scale
predicted_value = scaler.inverse_transform(prediction)

print(f"Predicted value: {predicted_value}")
```
## Output:
![Screenshot 2024-03-08 115012](https://github.com/NAVEENKUMAR4325/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119479566/c3f313e7-75b4-4b53-ae85-ea3bf78ad98f)


![Screenshot 2024-03-08 114703](https://github.com/NAVEENKUMAR4325/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119479566/d152e556-e5d8-4f4e-a301-a815fb5619de)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
