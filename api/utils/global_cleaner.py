from fastapi import HTTPException
from pandas import DataFrame
import pandas as pd

MAX_ROWS = 5000


def global_cleaner(df: DataFrame, target: str = None):
    clean_df = df.copy()

    # Create a compressed_df to use in tests: a dataframe of 5.000 rows maximum that maintains (in the best possible way) the same proportion as the original
    if len(clean_df) > MAX_ROWS:
        if target and target in clean_df.columns:
            value_counts = clean_df[target].value_counts()
            valid_classes = value_counts[value_counts > 1].index

            strat_df = clean_df[clean_df[target].isin(valid_classes)]

            clean_df = strat_df.groupby(target, group_keys=False).apply(
                lambda x: x.sample(
                    max(1, int(MAX_ROWS * len(x) / len(strat_df))), random_state=51
                )
            )

            clean_df = clean_df.sample(n=MAX_ROWS, random_state=51)

        else:
            clean_df = df.sample(n=MAX_ROWS, random_state=51)
            # Types normalization
    for col in clean_df.columns:
        if pd.api.types.is_object_dtype(clean_df[col]) or pd.api.types.is_string_dtype(
            clean_df[col]
        ):
            clean_df[col] = (
                clean_df[col].astype(str).str.strip()
            )  # Remove invisible spaces in str columns

            if clean_df[col].nunique() == 2:
                first_value = clean_df[col].unique()[0]
                clean_df[col] = (clean_df[col] == first_value).astype(
                    int
                )  # If just 2 possible values, transform in a binary column (1 and 0s)
        elif pd.api.types.is_numeric_dtype(clean_df[col]):
            clean_df[col] = pd.to_numeric(
                clean_df[col], errors="coerce"
            )  # Try to convert "number-string" columns to numeric

    # Target validation
    if target:
        if (
            clean_df[target].nunique() == 1
        ):  # Raises a error if the target just had a possible value.
            raise HTTPException(
                status_code=400,
                detail="Target needs to had at least 2 possible values.",
            )

        # Exclude rows where the target is null
        clean_df = clean_df[clean_df[target].notna()]

    # Remove constant columns
    constant_columns = [
        column for column in clean_df.columns if clean_df[column].nunique() == 1
    ]
    if constant_columns:
        clean_df.drop(columns=constant_columns, inplace=True)

    # Remove duplicates
    clean_df.drop_duplicates(inplace=True)  # Remove all duplicated rows

    # Remove columns with >60% missing values
    much_missing_values_columns = [
        column for column in clean_df.columns if clean_df[column].isna().mean() > 0.6
    ]
    if much_missing_values_columns:
        clean_df.drop(columns=much_missing_values_columns, inplace=True)

    return clean_df
