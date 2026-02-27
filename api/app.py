
from io import BytesIO
from pathlib import Path
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Form, UploadFile
import pandas as pd

from model_tests import check_colinearity

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.post("/analyze")
async def analyze(file: UploadFile, id_column: Optional[str] = Form(None), separator: Optional[str] = Form(None), sheet_name: Optional[str] = Form(None)):
    # File reader
    file_extension = Path(file.filename).suffix.lower()
    contents = await file.read()
    df = convert_to_df(BytesIO(contents), file_extension, sheet_name=sheet_name, sep=separator)
    
    # Delete id_column
    if id_column and id_column in df.columns.tolist():
        df.drop(columns=id_column, inplace=True)
        
    # Convert the numpy classes to primitive classes and return to the frontend
    converted_df = df.convert_dtypes()
    data = converted_df.to_dict(orient="records")
    
    # (column_name, column_primitive_type, example)
    return [(key, str(type(value)), value) for key, value in data[0].items()]

def convert_to_df(content: BytesIO, file_extension: str, **kwargs):    
    if file_extension == ".csv":
        df = pd.read_csv(content)
    
    elif file_extension in [".tsv", "txt"]:
        df = pd.read_csv(content, sep=kwargs.get("sep", "\t"))
    
    elif file_extension in [".xlsx", ".xlsm"]:
        df = pd.read_excel(content, engine="openpyxl", sheet_name=kwargs.get("sheet_name")) 
    
    elif file_extension == ".xls":
        df = pd.read_excel(content, engine="xlrd", sheet_name=kwargs.get("sheet_name"))
    
    elif file_extension == ".xlsb":
        df = pd.read_excel(content, engine="pyxlsb", sheet_name=kwargs.get("sheet_name"))
        
    elif file_extension == "ods":
        df = pd.read_excel(content, engine="odf", sheet_name=kwargs.get("sheet_name"))
    
    elif file_extension == ".json":
        df = pd.read_json(content)
    
    else:
        raise ValueError(f"Format not found: {file_extension}")

    return df