import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def optimize_hyperparameters(model, param_grid, X_train, y_train, search_type='grid', n_iter=10):
    if search_type == 'grid':
        search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    elif search_type == 'random':
        search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=5, scoring='neg_mean_squared_error', random_state=42)
    else:
        raise ValueError("search_type must be either 'grid' or 'random'")
    
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, -search.best_score_

def main(X_train, y_train):
    # Define models and hyperparameter grids
    models = {
        'Linear Regression': (LinearRegression(), {
            'fit_intercept': [True, False],
            'normalize': [True, False]
        }),
        'Logistic Regression': (LogisticRegression(max_iter=1000), {
            'C': np.logspace(-4, 4, 10),
            'solver': ['liblinear', 'saga'],
            'penalty': ['l1', 'l2']
        })
    }
    
    results = {}
    
    for model_name, (model, param_grid) in models.items():
        best_model, best_params, best_score = optimize_hyperparameters(model, param_grid, X_train, y_train, search_type='grid')
        results[model_name] = {
            'best_model': best_model,
            'best_params': best_params,
            'best_score': best_score
        }
    
    return results

# Example usage:
# X_train, y_train = ... # Load and preprocess your data
# results = main(X_train, y_train)
# print(results)