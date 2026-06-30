"""
1. importing necessaries libraries
2. Loading data by importing the load_data function from preprocessed.py
3. Create a custom transformer for high skewed numerical feaatures (yeo-johnson transformaation)
4. Create a custom transformer for selective one-hot-encoding.
5. Create a custom transformer for robust scaling (RobustScaler)
6. Create a function for data splitting and model training.
7. Complete preprocessing + Model Pipeline  
"""
# ===========================================
# SECTION 1: IMPORTING NECESSARY LIBRARIES
# ===========================================
import os
from pathlib import Path
import numpy as np
import pandas as pd
from src.preprocessed import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)
from src.customcolumntransformer import CustomColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PowerTransformer,
    RobustScaler,
    OneHotEncoder
)
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

# ===========================================
# SECTION 2: DATA LOADING
# ===========================================
data = load_data("data/processed/Telco-Customer-Churn-Cleaned.csv")

# =================================================================================
# SECTION 3: CREATING CUSTOM COLUMN TRANSFORMER FOR HIGH SKEWED NUMERICAL FEATURE
# =================================================================================
High_Skewed_Transformer = CustomColumnTransformer(
    columns=["total_charges"],
    transformer=PowerTransformer(method='yeo-johnson') 
)

# =================================================================================
# SECTION 4: CREATING CUSTOM COLUMN TRANSFORMER FOR SELECTIVE ONE-HOT-ENCODING
# =================================================================================
Categorical_Encoding = CustomColumnTransformer(
    columns=[
       "multiple_lines", "internet_service", "online_security", "online_backup", "device_protection",
       "tech_support", "streaming_tv", "streaming_movies", "contract", "payment_method" 
    ],
    transformer=OneHotEncoder
)

# =================================================================================
# SECTION 5: CREATING CUSTOM COLUMN TRANSFORMER FOR SELECTIVE ROBUST SCALER
# =================================================================================
Robust_Scaling = CustomColumnTransformer(
    columns=["tenure", "monthly_charges", "total_charges"],
    transformer=RobustScaler
)

# =================================================================================
# SECTION 6: CREATING FUNCTION FOR DATA SPLITTING AND MODEL TRAINING
# =================================================================================
def train_model(df:pd.DataFrame, target_col:str):
    # splitting df into feature matrix(X) and target variable(y)
    X = df.drop(columns=[target_col], axis="columns")
    y = df[target_col]
    
    # Splitting data into training set and validation set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Calculating scale_pos_weight for imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Creating an instance of XGBCLASSIFIER
    xgb = XGBClassifier(
        n_estimators = 500,
        learning_rate = 0.05,
        max_depth = 6,
        subsample = 0.8,
        colsample_bytree = 0.8,
        random_state = 42,
        n_jobs = -1,
        scale_pos_weight = scale_pos_weight,
        eval_metric = "logloss"
    )
    
    return X_train, X_test, y_train, xgb

# ======================================================
# SECTION 7: COMPLETE PREPROCESSING + MODEL PIPELINE
# ======================================================
pipeline = Pipeline(steps=[
    ("high_skewed", High_Skewed_Transformer),
    ("categorical", Categorical_Encoding),
    ("robust_scaling", Robust_Scaling),
    ("classifier", xgb_model)
])

# Fitting the pipeline to the training data
pipeline.fit(X_train, y_train)

# Making prediction 
proba = pipeline.predict_proba(X_test)[:,1]
