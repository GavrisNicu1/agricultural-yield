import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    # Assuming the target variable is named 'yield'
    X = df.drop('yield', axis=1)
    y = df['yield']
    return X, y

def train_logistic_regression(X_train, y_train, hyperparameters):
    model = LogisticRegression()
    grid_search = GridSearchCV(model, hyperparameters, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, report, cm

if __name__ == "__main__":
    # Load and preprocess data
    df = load_data('../data/crop_yield.csv')
    X, y = preprocess_data(df)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define hyperparameters for tuning
    hyperparameters = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    }

    # Train the model
    model = train_logistic_regression(X_train, y_train, hyperparameters)

    # Evaluate the model
    accuracy, report, cm = evaluate_model(model, X_test, y_test)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)