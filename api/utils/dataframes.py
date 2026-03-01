from io import BytesIO
import os

from dotenv import load_dotenv
import pandas as pd
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from extractors import extract_correlation_pairs


def apply_pca(
    X_transformed, df_transformed: DataFrame
):  # df_transformed = df after the preprocessiing
    base_pca = PCA()
    base_pca.fit(X_transformed)
    explained_variance = base_pca.explained_variance_ratio_.cumsum()
    n_components = (explained_variance < 0.9).sum() + 1
    model_pca = PCA(n_components=n_components)
    X_pca = model_pca.fit_transform(X_transformed)

    corr_pairs = extract_correlation_pairs(df_transformed)
    df_all_correlations = pd.DataFrame(corr_pairs)
    df_high_corr = df_all_correlations[abs(df_all_correlations["correlation"]) >= 0.6]
    return df_all_correlations, df_high_corr, n_components, X_pca


def make_preprocessor(
    numericals: list = [], categoricals: list = [], ordinals: list = []
):
    list_transformers = []
    if numericals:
        list_transformers.append(("num", StandardScaler(), numericals))
    if categoricals:
        list_transformers.append(("cat", OneHotEncoder(), categoricals))
    if ordinals:
        list_transformers.append(("ord", OrdinalEncoder(), ordinals))
    if list_transformers:
        preprocessor = ColumnTransformer(transformers=list_transformers)
        return preprocessor
    return None


def return_prediction(target: str, df: DataFrame, dict_values: dict, best_model):
    X = df.drop(columns=target)
    y = df[target]
    best_model.fit(X, y)
    to_predict_df = DataFrame([dict_values], columns=X.columns)
    y_predict = best_model.predict(to_predict_df)
    to_predict_df[target] = y_predict
    return to_predict_df


def compact_file_to_less_than_max_size_mb(df: DataFrame):
    # Return a df with 10% less data.
    return df.sample(frac=0.9, random_state=51)
