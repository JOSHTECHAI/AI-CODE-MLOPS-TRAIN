# Importing necessary libraries
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

# 
class CustomColumnTransformer(BaseEstimator, TransformerMixin):
    """
    Apply a scikit-learn transformer to selected columns of a DataFrame
    while leaving all other columns unchanged.

    Parameters
    ----------
    columns : list[str]
        List of columns to which the transformer should be applied.

    transformer : object
        Any scikit-learn transformer implementing fit() and transform().

    Attributes
    ----------
    feature_names_out_ : list[str]
        Names of the transformed columns after fitting.
    """
    
    def __init__(self, columns: list[str], transformer):
        self.columns = columns
        self.transformer = transformer

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer on the selected columns.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.

        y : array-like, optional
            Target variable (ignored).

        Returns
        -------
        CustomColumnTransformer
            Fitted transformer.
        """
        self.transformer.fit(X[self.columns])

        # Obtaining output feature names
        if hasattr(self.transformer, "get_feature_names_out"):
            self.feature_names_out_ = (self.transformer.get_feature_names_out(self.columns).tolist())
        else:
            self.feature_names_out_ = self.columns

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the selected columns and return a DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing transformed columns and untouched columns.
        """
        X = X.copy()

        # Applying transformation
        transformed = self.transformer.transform(X[self.columns])

        # Converting transformed output to DataFrame
        transformed_df = pd.DataFrame(
            transformed,
            columns=self.feature_names_out_,
            index=X.index
        )

        # Removing original columns
        X = X.drop(columns=self.columns)

        # Concatenating transformed columns with remaining columns
        X = pd.concat([X, transformed_df], axis=1)

        return X