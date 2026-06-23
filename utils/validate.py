"""
1. Creating function for data validation
2. Schema Definition
3. Schema Validation
4. Validation Result    
"""
# Importing necessary libraries for data validation
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import DataFrameSchema, Column, Check
from pandera.errors import SchemaErrors
from typing import Tuple, List
from preprocessed import clean_total_charges

# Function for Data Validation
def validate_data(df:pd.DataFrame) -> Tuple[bool, List[str]]:
    print("Starting data validation with Pandera.........")
    failed_checks = []
    
    # Schema definition
    # Cleaning the total charges column
    df = clean_total_charges(df)
    schema = DataFrameSchema(
        {
            # Customer identifier
            "customerID": Column(str, nullable=False, unique=True),
            # Demographic features
            "gender": Column(str, checks=Check.isin(["Male","Female"])),
            "Partner": Column(str, checks=Check.isin(["Yes","No"])),
            "Dependents": Column(str, checks=Check.isin(["Yes","No"])),
            "SeniorCitizen": Column(int, checks=Check.isin([0,1])),
        }
    )
    