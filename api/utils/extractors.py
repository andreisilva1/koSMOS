import pandas as pd


def extract_correlation_pairs(df: pd.DataFrame):
    corr_pairs = []
    cols = df.columns
    corr = df.corr()
    corr_pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_pairs.append(
                {"x": cols[i], "y": cols[j], "correlation": corr.iloc[i, j]}
            )
    return corr_pairs


def extract_numericals_categoricals_and_ordinals(dict_types: dict):
    numericals = [
        key
        for key in dict_types.keys()
        if dict_types[key]["col_type"] in ["range", "int", "float"]
    ]
    categoricals = [
        key
        for key in dict_types.keys()
        if dict_types[key]["col_type"] in ["enum", "str"]
    ]
    ordinals = [
        key for key in dict_types.keys() if dict_types[key]["col_type"] == "ordinal"
    ]
    return numericals, categoricals, ordinals
