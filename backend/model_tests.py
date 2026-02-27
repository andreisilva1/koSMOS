import numpy as np
from pandas import DataFrame
import pandas as pd
from pandas.api.types import is_string_dtype
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

def check_colinearity(df: DataFrame, target: str):
    # By Correlation Matriz
    corr = df.corr(numeric_only=True)
    pair_corr = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pair_corr.append(cols[i], cols[j], cols.iloc[i, j])
    
    # 1 pair with more than 0.9 correlation or 10% or the pairs more than 0.85 correlation = strong colinearity, that makes sense... I think      
    corr_more_than_90 = []
    corr_more_than_90.append([f1, f2, val] for f1, f2, val in pair_corr if abs(pair_corr[2]) >= 0.9)
    corr_more_than_85 = []
    corr_more_than_85.append([f1, f2, val] for f1, f2, val in pair_corr if abs(pair_corr[2]) >= 0.85)

    # Check colinearity by VIF
    X = df.drop(columns=target)
    
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
    
    # VIF Verification (Variance Inflation Factor) -> How much a variable is been explained by the others
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    # Numeric stability of the X matriz - Detect when small variations in data can cause large changes in coefficients
    cond = np.linalg.cond(StandardScaler().fit_transform(X))
    
    # Return true if one correlation > 0.9, 2 correlations > 0.85, VIF average >= 5, any VIF >= 10, cond > 100
    return len(corr_more_than_90) >= 1 or len(corr_more_than_85) / len(pair_corr) * 100 > 0.1 or vif_data["VIF"] / len(vif_data) >= 5 or vif_data[vif_data["VIF"] >= 10] or cond > 100
    