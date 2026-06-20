"""
1. Creating a function to load our data
2. Creating a function for data standardization
3. Creating a function to clean total charge column
4. Creating a function to identify categorical column
5. Creating a function to identify numerical column
6. Creating a function to perform binary encoding on binary columns (columns with cardinality of 2)
7. Creating a function to save the preprocess data inside the processed sub-folder in the data folder
"""
# Importing Necessary Libraries For Data Preprocessing
import pandas as pd
import numpy as np
import os

# Function for loading raw data into a pandas dataframe
def load_data(file_path:str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

# Function for standardizing column names, removing leading and trailing whitespaces and dropping the customer_id
def standardize_columns(df:pd.DataFrame) -> pd.DataFrame:
    standardize = {
        'customerID': 'customer_id',
        'gender': 'gender',
        'SeniorCitizen': 'senior_citizen',
        'Partner': 'partner',
        'Dependents': 'dependents',
        'tenure': 'tenure',
        'PhoneService': 'phone_service',
        'MultipleLines': 'multiple_lines',
        'InternetService': 'internet_service',
        'OnlineSecurity': 'online_security',
        'OnlineBackup': 'online_backup',
        'DeviceProtection': 'device_protection',
        'TechSupport': 'tech_support',
        'StreamingTV': 'streaming_tv',
        'StreamingMovies': 'streaming_movies',
        'Contract': 'contract',
        'PaperlessBilling': 'paperless_billing',
        'PaymentMethod': 'payment_method',
        'MonthlyCharges': 'monthly_charges',
        'TotalCharges': 'total_charges',
        'Churn': 'churn'
    }
    # creating a copy of the original dataframe
    df = df.copy()
    # renaming the columns names
    df.rename(columns = standardize, inplace = True)
    # dropping the customer_id
    df.drop(columns="cutomer_id", axis="columns", inplace=True)
    # Removing the leading and trailing whitespaces from string columns
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df[col] = df[col].str.strip()
        
    return df

# Function for cleaning and imputing the total_charges column
def clean_total_charges(df:pd.DataFrame) -> pd.DataFrame:
    # creating a copy of df
    df = df.copy()
    # identifying blank values in the total_charges column
    blank_values = df["total_charges"].astype(str).str.strip().eq("")
    # converting invalid values to NaN
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors = "coerce")
    # replace missing values with 0 for customers whose tenure is zero
    df.loc[(df['total_charges'].isna()) & (df['tenure']==0), 'total_charges'] = 0
    
    return df

# Function for getting categorical columns in the data
def get_categorical_columns(df:pd.DataFrame, max_unique_values:int = 10) -> list[str]:
    categorical_cols = []
    for col in df.columns:
        if(
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
            or pd.api.types.is_categorical_dtype(df[col])
            or df[col].nunique <= max_unique_values
        ):
            df[col] = df[col].astype("category")
            categorical_cols.append(col)
    
    return categorical_cols

# Function for getting numerical columns in the data
def get_numeric_columns(df:pd.DataFrame) -> list[str]:
    numeric_cols = (df.select_dtypes(include=["int64", "float64"]).columns.tolist())
    return numeric_cols 