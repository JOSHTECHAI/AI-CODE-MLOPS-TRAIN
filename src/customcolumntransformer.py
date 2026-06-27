"""
1. creating the CustomColumnTransformer class  
2. creating the constructor
3. defining class attributes
4. defining method(fit, transfrom)  
"""

# Importing necessaries libraries
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Creating the CustomColumnTransformer class
class CustomColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns:list[str], transformer):
        self.columns = columns
        self.transformer = transformer
    
    # Creating the fit method
    def fit(self, X:pd.DataFrame, y = None):
        self.transformer.fit(X[self.columns])
        # obtaining output features name
        if hasatrr(self.transformer, "get_feature_names_out"):
            self.feature_names_out_ = (self.transformer.get_feature_names_out(self.columns).tolist())
        else:
            self.feature_names_out_ = self.columns
        return self
    
    # Creating the transform method
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        transformed = self.transformer.transform(X[self.columns])
        # Converting transform output to dataframe
        transformed_df = pd.DataFrame(transformed, columns=self.feature_names_out_, index = X.index)
        # Removing originl columns
        X = X.drop(columns=self.columns)
        # Contenating the transform columns with the remaining columns
        X = pd.concat([X, transformed_df])
        return X