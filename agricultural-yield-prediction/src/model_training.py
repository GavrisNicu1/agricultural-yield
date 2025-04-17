import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import json

# Load the dataset
df = pd.read_csv('../data/crop_yield.csv')

# Preprocess the data (assuming preprocessing functions are defined in data_preprocessing.py)
from src.data_preprocessing import preprocess_data

X, y = preprocess_data(df)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
linear_model = LinearRegression()
logistic_model = LogisticRegression(max_iter=1000)

# Train Linear Regression model
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Train Logistic Regression model (assuming y is binary for classification)
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)

# Evaluate models
linear_mse = mean_squared_error(y_test, y_pred_linear)
logistic_accuracy = accuracy_score(y_test, y_pred_logistic)

# Hyperparameter optimization for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_logistic_model = grid_search.best_estimator_
y_pred_best_logistic = best_logistic_model.predict(X_test)
best_logistic_accuracy = accuracy_score(y_test, y_pred_best_logistic)

# Save model metrics to JSON
model_metrics = {
    'Linear Regression': {
        'Mean Squared Error': linear_mse
    },
    'Logistic Regression': {
        'Accuracy': logistic_accuracy,
        'Best Accuracy': best_logistic_accuracy,
        'Best Parameters': grid_search.best_params_
    }
}

with open('../results/model_metrics.json', 'w') as f:
    json.dump(model_metrics, f)

# Print training results
print("Linear Regression Mean Squared Error:", linear_mse)
print("Logistic Regression Accuracy:", logistic_accuracy)
print("Best Logistic Regression Accuracy:", best_logistic_accuracy)
print("Best Parameters for Logistic Regression:", grid_search.best_params_)