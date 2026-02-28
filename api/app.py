from io import BytesIO
import json
import os
from pathlib import Path
import pickle
from typing import Optional
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
import joblib
import boto3
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans
from sklearn.pipeline import Pipeline
from checks import check_dict_values
from database.models import MLModel
from database.session import load_model_from_db, save_model
from model_tests.clustering import test_clustering_algorithms
from utils import convert_to_df, extract_numericals_categoricals_and_ordinals
import zipfile
from cryptography.fernet import Fernet
from sklearn.neighbors import KNeighborsClassifier

load_dotenv()

s3 = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")
DATASET_KEY = os.getenv("DATASET_KEY").encode()
fernet = Fernet(DATASET_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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
                      else [str(df[column].unique()[0].item() if hasattr(df[column].unique()[0], "item") else df[column].unique()[0])],
                      "type": "categorical" if (len(df[column].unique()) <= 3 and len(df) >= 1000) else str(type(data[0][column]))
        }
        for column in df.columns
    ]    
    
    # Will return something like {"column": "Acidity", "values": [-0.491590483], "type": "<class 'float'>"},
    # {"column": "Quality", "values": ["good", "bad"], "type": "<class 'str'>"}, if len(value > 1 and (in that case) < 3 ) it will change his type to ordinal in the frontend if any column with more than 1.000 values has just 3 unique_values (Probably will need more checks than that)
    return list_columns_and_values

@app.post("/test_model", include_in_schema=False)
async def test_models(dataset_file: UploadFile, dict_types: str = Form(), dict_values: str = Form(), ids_columns: str = Form(None), target: str = Form(None), n_groups: int = Form(None), sheet_name: Optional[str] = Form(None), separator: Optional[str] = Form(None)):
    dict_types, dict_values = json.loads(dict_types), json.loads(dict_values)
        
    file_extension = Path(dataset_file.filename).suffix.lower()

    contents = await dataset_file.read()
    df = convert_to_df(BytesIO(contents), file_extension, sheet_name=sheet_name, sep=separator)
    
    if ids_columns:
        df.drop(columns=[column for column in json.loads(ids_columns)], inplace=True)
        
    numericals, categoricals, ordinals = extract_numericals_categoricals_and_ordinals(dict_types)
    
    if target:
        check_dict_values(dict_types, dict_values)
    # No target AND n_groups? Hierarquical cluster, if n_groups, k-means, if target... Let's code yet.
    if not target:
        if not n_groups:
            cluster_method = "hierarquical"
        else:
            cluster_method = "k-means"
        
        df_with_clusters, high_correlations, all_correlations, best_model, pp = test_clustering_algorithms(cluster_method=cluster_method, df=df, dict_values=dict_values, n_groups=n_groups, numericals=numericals, categoricals=categoricals, ordinals=ordinals)
        ml_model = pickle.dumps(best_model)
        preprocessor = pickle.dumps(pp)
        
        # Put the final dataset, the high correlation, the all correlation dataset and the model itself in a zip
        output = BytesIO()
        with zipfile.ZipFile(output, "w") as z:
            z.writestr("dataset.csv", df_with_clusters)
            z.writestr("high_correlations.csv", high_correlations)
            z.writestr("all_correlations.csv", all_correlations)
            z.writestr(f"ml_model.pkl", ml_model)
            z.writestr(f"preprocessor.pkl", preprocessor)

            
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=export.zip"
            }
        )

@app.post("/send_model")
async def send_model(preprocessor_file: UploadFile, dataset_file: UploadFile, model_file: UploadFile, model_id: str = Form(None), dict_types: str = Form(None), target: str = Form(None)):
    # The bytes of the model will be save in the database
    model_contents = await model_file.read()
    dataset_content = await dataset_file.read()
    preprocessor_content = await preprocessor_file.read()
    
    enc_dataset = fernet.encrypt(dataset_content)
        
    s3.put_object(
        Body=enc_dataset,
        Bucket=BUCKET_NAME,
        Key=model_id
    )
    return await save_model(MLModel(name=model_id, preprocessor=preprocessor_content, model=model_contents, dict_types=dict_types, target=target if target else None))

@app.get("/load_model/{model_id}")
async def load_model(model_id: str):
    model = await load_model_from_db(model_id)
    if model:
        # Return the dict_types to the frontend (or API), to show the types of each data and how the backend expects the data.
        return json.loads(model["dict_types"])
    raise HTTPException(status_code=404, detail="No model found with the provided ID.")

@app.post("/predict")
async def predict(body: dict):
    model_id, dict_values = body["model_id"], body["values"]
    
    # Return the model from db by model_id
    model = await load_model_from_db(model_id)
    
    if model:
        dict_types = json.loads(model["dict_types"])
        target = model.get("target", None)
        # To later: what to do if have a target?
        if target:
            check_dict_values(dict_types, dict_values) 
        
        # If not target, will use KNeighbors to predict based on the dict_values and the dataset with the clusters saved.
        load_model = joblib.load(BytesIO(model["model"]))
        load_preprocessor = joblib.load(BytesIO(model["preprocessor"]))
        
        # If the model type is a Clustering, will use KNeighbors to predict the cluster of the new item
        if type(load_model) in [AgglomerativeClustering, BisectingKMeans]:
            
            kneighbors_model = Pipeline([
                ("preprocessor", load_preprocessor),
                ("classifier", KNeighborsClassifier())
            ])
            
            obj = s3.get_object(Bucket=BUCKET_NAME, Key=model_id)
            dataset_encrypted = obj["Body"].read()
            dataset_decrypted = fernet.decrypt(dataset_encrypted)
            
            df = convert_to_df(BytesIO(dataset_decrypted), ".csv")
            X, y = df.drop(columns="cluster"), df["cluster"]
            
            # Use the database with the cluster to train the KNeighbor model
            kneighbors_model.fit(X, y)
            to_predict_df = DataFrame([dict_values], columns=X.columns)
            y_predict = kneighbors_model.predict(to_predict_df)
            to_predict_df["cluster"] = y_predict
            
            final_df = pd.concat([df, to_predict_df], ignore_index=True)
            
            output = BytesIO()
            
            with zipfile.ZipFile(output, "w") as z:
                z.writestr("final_df.csv", final_df.to_csv(index=True))
                z.writestr("to_predict_df.csv", to_predict_df.to_csv(index=True))
            
            # Convert the df to csv -> convert the csv to bytes -> encrypt the bytes and save in S3
            buffer = BytesIO()
            final_df.to_csv(buffer, index=False)
            enc_final_df = fernet.encrypt(buffer.getvalue())
            
            s3.put_object(
            Body=enc_final_df,
            Bucket=BUCKET_NAME,
            Key=model_id
            ) 
            
            output.seek(0)

            # Will return a zip with the final_df (the dataset + the new prediction) + the dataset ONLY with the prediction result
            return StreamingResponse(output, media_type="application/zip", headers={"Content-Disposition": "attachment; export.zip"})
        
    raise HTTPException(status_code=404, detail="No model found with the provided ID.")