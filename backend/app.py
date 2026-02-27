
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, UploadFile
import pandas as pd
from pydantic import BaseModel

from model_tests import check_colinearity

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile, id_column: str = None):
    # File reader
    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))
    
    # Delete id_column
    if id_column and id_column in df.columns[0].split(","):
        df.drop(columns=id_column, inplace=True)
      
    return check_colinearity(df, "y")
