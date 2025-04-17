import pandas as pd
import numpy as np

def create_interaction_features(df):
    """
    Create interaction features from the existing dataset.
    """
    df['feature1_feature2'] = df['feature1'] * df['feature2']
    df['feature3_feature4'] = df['feature3'] * df['feature4']
    return df

def create_polynomial_features(df, degree=2):
    """
    Create polynomial features for numerical columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        for d in range(2, degree + 1):
            df[f'{col}_poly_{d}'] = df[col] ** d
    return df

def encode_categorical_features(df):
    """
    Encode categorical features using one-hot encoding.
    """
    df = pd.get_dummies(df, drop_first=True)
    return df

def feature_engineering(df):
    """
    Perform feature engineering on the dataset.
    """
    df = create_interaction_features(df)
    df = create_polynomial_features(df)
    df = encode_categorical_features(df)
    return df