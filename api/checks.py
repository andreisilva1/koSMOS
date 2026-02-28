from fastapi import HTTPException
import numpy as np
from pandas import DataFrame
import pandas as pd
from pandas.api.types import is_string_dtype
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

def check_collinearity(df: DataFrame, target: str = None):
    X = df.drop(columns=target) if target else df.copy()
    

    # Made a OneHot Encoder in all the str columns in the df
    str_columns = [X[col] for col in X.columns if is_string_dtype(X[col])]
    really_str_columns = []
    
    # Maybe a int/float saved as string? Better do a reconfirmation
    for col in str_columns:
        try:
            int(col)
        except:
            really_str_columns.append(col)
    
    df = pd.get_dummies(df, columns=really_str_columns, dtype="int64")
    
    # By Correlation Matriz
    corr = df.corr(numeric_only=True)
    print(df.corr())
    corr_pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_pairs.append({"col_1": cols[i], "col_2": cols[j], "corr": corr.iloc[i, j]})
    
    # 1 pair with more than 0.9 correlation or 10% or the pairs more than 0.85 correlation = strong collinearity, that makes sense... I think      
    corr_more_than_90 = []
    corr_more_than_85 = []

    corr_more_than_90 = len([x for x in corr_pairs if abs(x["corr"]) >= 0.9])
    corr_more_than_85 = len([x for x in corr_pairs if abs(x["corr"]) >= 0.85])

    # Check collinearity by VIF    
    # VIF Verification (Variance Inflation Factor) -> How much a variable is been explained by the others
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    # Numeric stability of the X matriz - Detect when small variations in data can cause large changes in coefficients
    cond = np.linalg.cond(StandardScaler().fit_transform(X))
    
    # Return true if one correlation > 0.9, 2 correlations > 0.85, VIF average >= 5, any VIF >= 10 or cond > 100
    return [bool(x) for x in [corr_more_than_90 >= 1, (corr_more_than_85 / len(corr_pairs)) > 0.1, (sum(vif_data.VIF) / len(vif_data)) >= 5, len(vif_data[vif_data["VIF"] >= 10]) > 0, cond > 100]]

def check_dict_values(dict_types: dict, dict_values: dict):
    error_list = []
    for key, value in dict_types.items():
        if not dict_values[key]:
            error_list.append(f"{key} needs to be provided.")
            continue
            
        if value["col_type"] in ["int", "float"] and type(dict_values[key]) == str:
            error_list.append(f"{key} needs to be integer.")
        
        if value["col_type"] in ["enum", "ordinal"] and type(dict_values[key] not in value["values"]):
            error_list.append(f"{key} needs to be one of the options: {value['values']}.")
            
        if value["col_type"] == "range" and type(dict_values[key] > value["values"][1] or dict_values[key] < value["values"][0]):
            error_list.append(f"{key} needs to be between: {value['values'][0]} - {value['values'][1]}.")
            
        if value["col_type"] == "range" and value["values"][2] == 1 and type(dict_values[key] == float):
            error_list.append(f"{key} needs to be a integer.")
    
    if error_list:
        raise HTTPException(status_code=400, detail={"error_list": error_list})
    