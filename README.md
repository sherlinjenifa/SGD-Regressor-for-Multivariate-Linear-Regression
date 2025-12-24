# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Input Data
- Collect dataset with features (independent variables) such as:
- Size of house (sq. ft.)
- Number of rooms
- Location index
- Age of house
- Amenities, etc.
- Target variables (dependent):
- Y1: Price of the house
- Y2: Number of occupants
- Step 2: Preprocess Data
- Handle missing values.
- Normalize/standardize features (important for SGD).
- Split dataset into training set and test set.
Step 3: Initialize Model
- Use SGDRegressor from scikit-learn.
- Choose hyperparameters:
- loss (e.g., squared_loss)
- penalty (e.g., l2 for regularization)
- learning_rate (constant, adaptive, or optimal)
- - max_iter (number of iterations)
Step 4: Train Model
- Fit the model on training data:
Y=W\cdot X+b- SGD updates weights iteratively:
W_{new}=W_{old}-\eta \cdot \nabla L(W)
- where \eta  = learning rate, L(W) = loss function.
Step 5: Predict- For new input features (house size, rooms, etc.):
- Predict price.
- Predict number of occupants.
- If predicting both simultaneously, use multi-output regression (wrap SGDRegressor with MultiOutputRegressor).
- - max_iter (number of iterations)
Step 4: Train Model
- Fit the model on training data:
Y=W\cdot X+b- SGD updates weights iteratively:
W_{new}=W_{old}-\eta \cdot \nabla L(W)
- where \eta  = learning rate, L(W) = loss function.
Step 5: Predict- For new input features (house size, rooms, etc.):
- Predict price.
- Predict number of occupants.
- If predicting both simultaneously, use multi-output regression (wrap SGDRegressor with MultiOutputRegressor).






## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: 
RegisterNumber:  
*/
```
#Manual Implementation using Numpy
import numpy as np

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
# Features: [Hours Studied, Attendance, Previous Marks]
X = np.array([
    [2, 80, 50],
    [3, 60, 40],
    [5, 90, 70],
    [7, 85, 80],
    [9, 95, 90]
], dtype=float)

# Target: Marks Scored
y = np.array([50, 45, 70, 80, 95], dtype=float)

# ------------------------------
# Step 2: Feature normalization
# ------------------------------
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# Add bias term (intercept)
X = np.c_[np.ones(X.shape[0]), X]  # shape becomes (n_samples, n_features + 1)

# ------------------------------
# Step 3: Initialize weights
# ------------------------------
n_features = X.shape[1]
weights = np.zeros(n_features)

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# ------------------------------
# Step 4: Stochastic Gradient Descent
# ------------------------------
for epoch in range(epochs):
    for i in range(X.shape[0]):
        xi = X[i]
        yi = y[i]
        y_pred = np.dot(xi, weights)
        error = y_pred - yi
        # Update weights
        weights -= learning_rate * error * xi

print("Trained Weights (including intercept):", weights)

# ------------------------------
# Step 5: Make predictions
# ------------------------------
y_pred_all = np.dot(X, weights)
print("Predicted values:", y_pred_all)

## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house](sam.png)
<img width="860" height="63" alt="image" src="https://github.com/user-attachments/assets/7b9fbddc-0dfb-4dd5-844c-39bba9c60af8" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
