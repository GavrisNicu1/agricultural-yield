import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    # Assuming the target variable is 'yield' and the rest are features
    X = df.drop(columns=['yield'])
    y = df['yield']
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    
    return model

def make_predictions(model, new_data):
    predictions = model.predict(new_data)
    return predictions

if __name__ == "__main__":
    filepath = '../data/crop_yield.csv'
    df = load_data(filepath)
    X, y = preprocess_data(df)
    model = train_model(X, y)