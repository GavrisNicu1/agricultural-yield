import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
import json

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

def compare_models(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Logistic Regression': LogisticRegression(max_iter=1000)
    }

    results = {}

    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        # Make predictions
        y_pred = model.predict(X_test)
        # Evaluate the model
        accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)
        results[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

    return results

def optimize_hyperparameters(X, y):
    # Example for Logistic Regression hyperparameter optimization
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    }
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
    grid_search.fit(X, y)
    return grid_search.best_params_

def save_results_to_json(results, filepath):
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    # Load data
    df = load_data('../data/crop_yield.csv')
    
    # Assume the last column is the target variable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Compare models
    results = compare_models(X, y)
    print(results)

    # Optimize hyperparameters for Logistic Regression
    best_params = optimize_hyperparameters(X, y)
    print(f"Best Hyperparameters for Logistic Regression: {best_params}")

    # Save results to JSON
    save_results_to_json(results, '../results/model_metrics.json')