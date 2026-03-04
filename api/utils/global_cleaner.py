from fastapi import HTTPException
from pandas import DataFrame
import pandas as pd

MAX_ROWS = 5000


def global_cleaner(df: DataFrame, target: str = None):
    clean_df = df.copy()

    changed_columns = {
        "changed_to_binary": [],
        "removed_by_unique_values": [],
        "removed_by_null_target": [],
        "removed_by_constant_value": [],
        "removed_by_many_missing_values": [],
        "probably_unique_values_columns": [],
    }
    probably_unique_values_columns = []
    for column in clean_df.columns:
        if clean_df[column].nunique() / len(clean_df) > 0.85:
            if not pd.api.types.is_numeric_dtype(clean_df[column]):
                clean_df.drop(
                    columns=column,
                    inplace=True,
                )
                probably_unique_values_columns.append(column)
    changed_columns["probably_unique_values_columns"] = probably_unique_values_columns
    # Drop columns with number of values = 85% of the number os rows (columns that MAYBE can be IDs)
    clean_df.drop(
        columns=probably_unique_values_columns,
        inplace=True,
    )

    changed_columns["removed_by_unique_values"] = probably_unique_values_columns

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
                changed_columns["changed_to_binary"].append(col)
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
        changed_columns["rows_removed_by_null_target"] = clean_df[
            clean_df[target].isna()
        ].sum()
        clean_df = clean_df[clean_df[target].notna()]

    # Remove constant columns
    constant_columns = [
        column for column in clean_df.columns if clean_df[column].nunique() == 1
    ]

    changed_columns["removed_by_constant_value"] = constant_columns
    if constant_columns:
        clean_df.drop(columns=constant_columns, inplace=True)

    # Remove duplicates
    clean_df.drop_duplicates(inplace=True)  # Remove all duplicated rows

    # Remove columns with >60% missing values
    much_missing_values_columns = [
        column for column in clean_df.columns if clean_df[column].isna().mean() > 0.6
    ]

    changed_columns["removed_by_many_missing_values"] = much_missing_values_columns

    if much_missing_values_columns:
        clean_df.drop(columns=much_missing_values_columns, inplace=True)

    not_null_changed_columns = {}
    for key, value in changed_columns.items():
        if value:
            not_null_changed_columns[key] = value

    changed_columns_df = None
    if not_null_changed_columns:
        changed_columns_df = DataFrame([not_null_changed_columns])

    return clean_df, changed_columns_df
