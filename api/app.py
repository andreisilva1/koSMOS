
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Optional
import boto3
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Form, UploadFile
from pandas import DataFrame

from model_tests.clustering import test_clustering_algorithms
from utils import convert_to_df
from checks import check_colinearity

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

s3 = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

@app.post("/analyze")
async def analyze(file: UploadFile, separator: Optional[str] = Form(None), sheet_name: Optional[str] = Form(None)):
    # File reader
    contents = await file.read()
    file_extension = Path(file.filename).suffix.lower()
    df = convert_to_df(BytesIO(contents), file_extension, sheet_name=sheet_name, sep=separator)
            
    # Convert the numpy classes to primitive classes to return to the frontend
    converted_df = df.convert_dtypes()
    data = converted_df.to_dict(orient="records")
    
    list_columns_and_values = [
        {
            "column": column,
            "values": [v.item() if hasattr(v, "item") else v for v in df[column].unique()]
                      if (len(df[column].unique()) <= 3 and len(df) >= 1000)
                      else [df[column].unique()[0].item()],
                      "type": "categorical" if (len(df[column].unique()) <= 3 and len(df) >= 1000) else str(type(data[0][column]))
        }
        for column in df.columns
    ]    
    
    # Will return something like {"column": "Acidity", "values": [-0.491590483], "type": "<class 'float'>"},
    # {"column": "Quality", "values": ["good", "bad"], "type": "<class 'str'>"}, if len(value > 1 and (in that case) < 3 ) it will change his type to ordinal in the frontend if any column with more than 1.000 values has just 3 unique_values (Probably will need more checks than that)
    return list_columns_and_values

@app.post("/test_model", include_in_schema=False)
async def test_models(file: UploadFile, file_extension: str, dict_types: str, dict_values: str, ids_columns: list = None, target: str = None, n_groups: int = None, sheet_name: Optional[str] = Form(None), separator: Optional[str] = Form(None)):
    dict_types, dict_values = json.loads(dict_types), json.loads(dict_values)
    file_extension = Path(file.filename).suffix.lower()

    contents = await file.read()
    df = convert_to_df(BytesIO(contents), file_extension, sheet_name=sheet_name, sep=separator)
    
    if ids_columns:
        df.drop(columns=ids_columns, inplace=True)
        
    numericals, categoricals, ordinals = extract_numericals_categoricals_and_ordinals(dict_types)
    
    # No target AND n_groups? Hierarquical cluster, if n_groups, k-means, if target... Let's code yet.
    if not target:
        if not n_groups:
            cluster_method = "hierarquical"
        else:
            cluster_method = "k-means"
            
    test_clustering_algorithms(cluster_method=cluster_method, df=df, dict_values=dict_values, n_groups=n_groups, numericals=numericals, categoricals=categoricals, ordinals=ordinals)


def extract_numericals_categoricals_and_ordinals(dict_types: dict):
    numericals = [key for key in dict_types.keys() if dict_types.get(key) in ["range", "int", "float"]]
    categoricals = [key for key in dict_types.keys() if dict_types.get(key) == "enum"]
    ordinals = [key for key in dict_types.keys() if dict_types.get(key) == "ordinal"]
    return numericals, categoricals, ordinals

