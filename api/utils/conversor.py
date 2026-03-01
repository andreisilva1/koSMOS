from io import BytesIO

import pandas as pd


def convert_to_df(content: BytesIO, file_extension: str, **kwargs):
    if file_extension == ".csv":
        df = pd.read_csv(content)

    elif file_extension in [".tsv", "txt"]:
        df = pd.read_csv(content, sep=kwargs.get("sep", "\t"))

    elif file_extension in [".xlsx", ".xlsm"]:
        df = pd.read_excel(
            content, engine="openpyxl", sheet_name=kwargs.get("sheet_name")
        )

    elif file_extension == ".xls":
        df = pd.read_excel(content, engine="xlrd", sheet_name=kwargs.get("sheet_name"))

    elif file_extension == ".xlsb":
        df = pd.read_excel(
            content, engine="pyxlsb", sheet_name=kwargs.get("sheet_name")
        )

    elif file_extension == "ods":
        df = pd.read_excel(content, engine="odf", sheet_name=kwargs.get("sheet_name"))

    elif file_extension == ".json":
        df = pd.read_json(content)

    else:
        raise ValueError(f"Format not found: {file_extension}")

    return df
