import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from .extractors import extract_correlation_pairs


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
    if len(numericals) > 0:
        list_transformers.append(("scaler", StandardScaler(), numericals))
    if len(categoricals) > 0:
        list_transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categoricals)
        )
    if len(ordinals) > 0:
        list_transformers.append(("ord", OrdinalEncoder(), ordinals))
    if len(list_transformers) > 0:
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


def return_accuracy_regression(y_pred, y_test):
    tolerancy = 0.10  # considers the prediction that “the error will be no more than 10% of the actual value” to be correct (the old formula returned extreme negative values, for a unknown reason)

    correct_guesses = np.abs((y_test - y_pred) / y_test) < tolerancy

    accuracy = np.mean(correct_guesses)
    return accuracy * 100


def return_accuracy_classification(y_pred, y_test):
    return (y_pred == y_test).mean() * 100
