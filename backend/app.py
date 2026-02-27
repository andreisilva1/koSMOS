
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, UploadFile
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

class BaseFile(BaseModel):
    file: UploadFile
    id_column: Optional[str]

@app.post("/analyze")
async def analyze(dict_file: BaseFile):
    # File reader
    contents = await dict_file.file.read()
    df = pd.read_table(BytesIO(contents))
    
    # Delete id column if user says
    if dict_file.id_column:
        df = df.drop(columns=dict_file.id_column)
