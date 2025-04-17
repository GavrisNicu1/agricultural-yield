import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Load the dataset from a CSV file.
    """
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """
    Clean the dataset by handling missing values and removing duplicates.
    """
    # Handle missing values
    df = df.dropna()  # Drop rows with missing values
    df = df.drop_duplicates()  # Remove duplicate rows
    return df

def preprocess_data(filepath):
    """
    Load and preprocess the data.
    """
    df = load_data(filepath)
    df = clean_data(df)
    return df

def encode_categorical_features(df):
    """
    Encode categorical features using one-hot encoding.
    """
    df = pd.get_dummies(df, drop_first=True)
    return df

def scale_features(df):
    """
    Scale numerical features using standard scaling.
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def prepare_data(filepath):
    """
    Complete preprocessing pipeline: load, clean, encode, and scale data.
    """
    df = preprocess_data(filepath)
    df = encode_categorical_features(df)
    df = scale_features(df)
    return df