from fastapi import HTTPException
from pandas import DataFrame
import pandas as pd

def global_cleaner(df: DataFrame, target: str = None):
    clean_df = df.copy()
    
    # Types normalization
    for col in clean_df.columns:
        if pd.api.types.is_object_dtype(clean_df[col]):
            clean_df[col] = clean_df[col].astype(str).str.strip() # Remove invisible spaces in str columns
        
        if pd.api.types.is_numeric_dtype(clean_df[col]):
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce") # Try to convert "number-string" columns to numeric
    
    # Target validation     
    if target: # Exclude rows where the target is null
        if clean_df[target].nunique() == 1:
            raise HTTPException(status_code=400, detail="Target needs to had at least 2 possible values.")
        clean_df = clean_df[clean_df[target].notna()]
    
    # Remove constant columns
    constant_columns = [column for column in clean_df.columns if clean_df[column].nunique() == 1]
    if constant_columns:
        clean_df.drop(columns=constant_columns, inplace=True)
    
    # Remove duplicates
    clean_df.drop_duplicates(inplace=True) # Remove all duplicated rows
    
    # Remove columns with >60% missing values
    much_missing_values_columns = [column for column in clean_df.columns if clean_df[column].isna().mean() > 0.6]
    if much_missing_values_columns:
        clean_df.drop(columns=much_missing_values_columns, inplace=True)
    
    return clean_df
    
    
