from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
import json
import os
from pathlib import Path
import pickle
from typing import Optional
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
import joblib
import boto3
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from checks.request import check_dict_values
from checks.database import check_aws_connection
from model_tests.regression import test_regression_algorithms
from database.models import MLModel
from database.session import ModelServiceDep, create_local_tables
from model_tests.clustering import test_clustering_algorithms
from utils import (
    compact_file_to_less_than_max_size_mb,
    convert_to_df,
    extract_numericals_categoricals_and_ordinals,
    return_prediction,
)
import zipfile
from cryptography.fernet import Fernet
from sklearn.neighbors import KNeighborsClassifier

load_dotenv()

MAX_POSSIBLE_SIZE_ORIGINAL_FILE = (
    200 * 1024 * 1024
)  # If the csv is less than 200mb, will compact him to ~25MB, otherwise, no chance...
MAX_SIZE_TO_SAVE = 25 * 1024 * 1024  # 25MB is the maximum size of test datasets for now
s3 = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

DATASET_KEY = os.getenv("DATASET_KEY").encode()
fernet = Fernet(DATASET_KEY)


@asynccontextmanager
async def lifespan_handler(app: FastAPI):
    if bool(os.getenv("ALLOW_LOCAL_FALLBACK")):
        await create_local_tables()
    yield


app = FastAPI(lifespan=lifespan_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze", include_in_schema=False)
async def analyze(
    file: UploadFile,
    separator: Optional[str] = Form(None),
    sheet_name: Optional[str] = Form(None),
):
    # File reader
    contents = await file.read()

    if len(contents) > MAX_POSSIBLE_SIZE_ORIGINAL_FILE:
        raise HTTPException(
            status_code=400,
            detail=f"The file size exceeds our safe limit ({MAX_SIZE_TO_SAVE / 1024 / 1024:.2f}mb) and our compression limit ({MAX_POSSIBLE_SIZE_ORIGINAL_FILE / 1024 / 1024:.2f}mb).",
        )

    file_extension = Path(file.filename).suffix.lower()
    df = convert_to_df(
        BytesIO(contents), file_extension, sheet_name=sheet_name, sep=separator
    )

    df = convert_to_df(
        BytesIO(contents), file_extension, sheet_name=sheet_name, sep=separator
    )

    df_size_in_mb = df.memory_usage(deep=True).sum() / (1024**2)

    while df_size_in_mb > MAX_SIZE_TO_SAVE:
        # Return a df with 10% less data until it stops exceeding the safe backup limit on S3
        df = compact_file_to_less_than_max_size_mb(df)

    # Convert the numpy classes to primitive classes to return to the frontend
    converted_df = df.convert_dtypes()
    data = converted_df.to_dict(orient="records")

    list_columns_and_values = [
        {
            "column": column,
            "values": (
                [v.item() if hasattr(v, "item") else v for v in df[column].unique()]
                if (len(df[column].unique()) <= 3 and len(df) >= 1000)
                else [
                    str(
                        df[column].unique()[0].item()
                        if hasattr(df[column].unique()[0], "item")
                        else df[column].unique()[0]
                    )
                ]
            ),
            "type": (
                "categorical"
                if (len(df[column].unique()) <= 3 and len(df) >= 1000)
                else str(type(data[0][column]))
            ),
        }
        for column in df.columns
    ]

    # Will return something like {"column": "Acidity", "values": [-0.491590483], "type": "<class 'float'>"},
    # {"column": "Quality", "values": ["good", "bad"], "type": "<class 'str'>"}, if len(value > 1 and (in that case) < 3 ) it will change his type to ordinal in the frontend if any column with more than 1.000 values has just 3 unique_values (Probably will need more checks than that)
    return list_columns_and_values


@app.post("/test_model", include_in_schema=False)
async def test_models(
    dataset_file: UploadFile,
    dict_types: str = Form(),
    dict_values: str = Form(),
    ids_columns: str = Form(None),
    target: str = Form(None),
    n_groups: int = Form(None),
    sheet_name: Optional[str] = Form(None),
    separator: Optional[str] = Form(None),
):
    contents = await dataset_file.read()

    dict_types, dict_values = json.loads(dict_types), json.loads(dict_values)

    file_extension = Path(dataset_file.filename).suffix.lower()

    df = convert_to_df(
        BytesIO(contents), file_extension, sheet_name=sheet_name, sep=separator
    )

    if ids_columns:
        df.drop(columns=[column for column in json.loads(ids_columns)], inplace=True)

    numericals, categoricals, ordinals = extract_numericals_categoricals_and_ordinals(
        dict_types
    )

    if target:
        check_dict_values(dict_types, dict_values)

        # Check if target is numeric or categoric
        if pd.api.types.is_numeric_dtype(df[target]):
            best_model, pp, accuracy, csv_high_correlations, csv_all_correlations = (
                test_regression_algorithms(
                    target=target,
                    df=df,
                    numericals=numericals,
                    categoricals=categoricals,
                    ordinals=ordinals,
                )
            )

            ml_model = pickle.dumps(best_model)
            preprocessor_pkl = pickle.dumps(pp)

            X = df.drop(columns=target)
            y = df[target]

            X_transformed = pp.fit_transform(X)

            best_model.fit(X_transformed, y)
            prediction_df = DataFrame([dict_values], columns=X.columns)
            dict_values_transformed = pp.transform(prediction_df)

            y_predict = best_model.predict(dict_values_transformed)
            prediction_df[target] = y_predict

            accuracy_df = DataFrame(
                [{"accuracy": f"{accuracy:.2f}"}], columns=["accuracy"]
            )

            # Put the final dataset, the high correlation, the all correlation dataset and the model itself in a zip
            output = BytesIO()

            df_with_prediction = pd.concat([df, prediction_df], ignore_index=True)
            with zipfile.ZipFile(output, "w") as z:
                z.writestr("dataset.csv", df_with_prediction.to_csv(index=False)),
                z.writestr("prediction.csv", prediction_df.to_csv(index=False)),
                z.writestr("accuracy.csv", accuracy_df.to_csv(index=False)),
                z.writestr("high_correlations.csv", csv_high_correlations)
                z.writestr("all_correlations.csv", csv_all_correlations)
                z.writestr(f"ml_model.pkl", ml_model)
                z.writestr(f"preprocessor.pkl", preprocessor_pkl)

            output.seek(0)
    if not target:
        if not n_groups:
            # No target AND n_groups? Hierarquical cluster.
            cluster_method = "hierarquical"
        else:
            # No target WITH n_groups? K-means.
            cluster_method = "k-means"

        (
            df_with_clusters,
            csv_high_correlations,
            csv_all_correlations,
            best_model,
            pp,
        ) = test_clustering_algorithms(
            cluster_method=cluster_method,
            df=df,
            n_groups=n_groups,
            numericals=numericals,
            categoricals=categoricals,
            ordinals=ordinals,
        )
        ml_model = pickle.dumps(best_model)
        preprocessor = pickle.dumps(pp)

        # Put the final dataset, the high correlation, the all correlation dataset and the model itself in a zip
        output = BytesIO()
        with zipfile.ZipFile(output, "w") as z:
            z.writestr("dataset.csv", df_with_clusters)
            z.writestr("high_correlations.csv", csv_high_correlations)
            z.writestr("all_correlations.csv", csv_all_correlations)
            z.writestr(f"ml_model.pkl", ml_model)
            z.writestr(f"preprocessor.pkl", preprocessor)

        output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=export.zip"},
    )


@app.post("/send_model", include_in_schema=False)
async def send_model(
    service: ModelServiceDep,
    preprocessor_file: UploadFile,
    dataset_file: UploadFile,
    model_file: UploadFile,
    model_id: str = Form(None),
    dict_types: str = Form(None),
    target: str = Form(None),
):
    # The bytes of the model will be save in the database
    model_contents = await model_file.read()
    dataset_content = await dataset_file.read()
    preprocessor_content = await preprocessor_file.read()

    enc_dataset = fernet.encrypt(dataset_content)

    is_aws_connected = check_aws_connection(
        s3, BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_DEFAULT_REGION, AWS_SECRET_ACCESS_KEY
    )
    if is_aws_connected:
        s3.put_object(Body=enc_dataset, Bucket=BUCKET_NAME, Key=model_id)

    else:
        os.makedirs("encrypted_datasets", exist_ok=True)
        with open(f"encrypted_datasets/{model_id}", "wb") as f:
            f.write(enc_dataset)
    await service.save_model(
        MLModel(
            name=model_id,
            preprocessor=preprocessor_content,
            model=model_contents,
            dict_types=dict_types,
            target=target if target else None,
            created_at=datetime.now(),
        )
    )
    return JSONResponse(status_code=200, content={"detail": "Dataset saved."})


@app.get("/load_model/{model_id}")
async def load_model(service: ModelServiceDep, model_id: str):
    model = await service.load_model_from_db(model_id)
    if model:
        # Return the dict_types to the frontend (or API), to show the types of each data and how the backend expects the data.
        return json.loads(model["dict_types"])
    raise HTTPException(status_code=404, detail="No model found with the provided ID.")


@app.post("/predict/{model_id}")
async def predict(
    service: ModelServiceDep, model_id: str, dict_values: dict = Body(None)
):

    # Return the model from db by model_id
    model = await service.load_model_from_db(model_id)

    if model:
        dict_types = json.loads(model["dict_types"])
        if any(key not in dict_values.keys() for key in dict_types.keys()):
            raise HTTPException(
                status_code=400,
                detail=f"The dict send is not compatible with this model. Expected dict: {dict_types}",
            )
        is_aws_connected = check_aws_connection(
            s3,
            BUCKET_NAME,
            AWS_ACCESS_KEY_ID,
            AWS_DEFAULT_REGION,
            AWS_SECRET_ACCESS_KEY,
        )

        # If connected with S3, take the file from there, if not AND ALLOW_FALLBACK is activated, take from encrypted_datasets folder.
        if is_aws_connected:
            obj = s3.get_object(Bucket=BUCKET_NAME, Key=model_id)
            dataset_encrypted = obj["Body"].read()

        else:
            with open(f"encrypted_datasets/{model_id}", "rb") as f:
                dataset_encrypted = f.read()

        dataset_decrypted = fernet.decrypt(dataset_encrypted)

        df = convert_to_df(BytesIO(dataset_decrypted), ".csv")

        output = BytesIO()
        load_model = joblib.load(BytesIO(model["model"]))
        load_preprocessor = joblib.load(BytesIO(model["preprocessor"]))

        check_dict_values(dict_types, dict_values)

        target = model.get("target", None)

        # If there is a target, it will load the exact model saved in the database,
        # since it does not depend on what was saved in the original dataset for that, only on the training done on that dataset.
        if target:
            model = load_model
            X = df.drop(columns=target)
            y = df[target]

            X_transformed = load_preprocessor.fit_transform(X)

            model.fit(X_transformed, y)
            prediction_df = DataFrame([dict_values], columns=X.columns)
            dict_values_transformed = load_preprocessor.transform(prediction_df)

            y_predict = model.predict(dict_values_transformed)
            prediction_df[target] = y_predict

        # If no target, will use KNeighbors and the dataset with clusters to determine the new item cluster (it depends on the cluster division).
        else:
            model = Pipeline(
                [
                    ("preprocessor", load_preprocessor),
                    ("classifier", KNeighborsClassifier()),
                ]
            )

            # Use the database with the cluster to train the KNeighbor model and return the prediction_df
            prediction_df = return_prediction(
                target="cluster", df=df, dict_values=dict_values, best_model=model
            )

        final_df = pd.concat([df, prediction_df], ignore_index=True)

        # Convert the df to csv -> convert the csv to bytes -> encrypt the bytes and save in S3
        buffer = BytesIO()
        final_df.to_csv(buffer, index=False)
        enc_final_df = fernet.encrypt(buffer.getvalue())

        # If AWS configuration, use S3, else, fallback to local
        is_aws_connected = check_aws_connection(
            s3,
            BUCKET_NAME,
            AWS_ACCESS_KEY_ID,
            AWS_DEFAULT_REGION,
            AWS_SECRET_ACCESS_KEY,
        )
        if is_aws_connected:
            s3.put_object(Body=enc_final_df, Bucket=BUCKET_NAME, Key=model_id)

        else:
            os.makedirs("encrypted_datasets", exist_ok=True)
            with open(f"encrypted_datasets/{model_id}", "wb") as f:
                f.write(enc_final_df)

        with zipfile.ZipFile(output, "w") as z:
            z.writestr("to_predict_df.csv", prediction_df.to_csv(index=False)),
            z.writestr("final_df.csv", final_df.to_csv(index=False)),

        output.seek(0)

        # Will return a zip with the final_df (the dataset + the new prediction) + the dataset ONLY with the prediction result
        return StreamingResponse(
            output,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; export.zip"},
        )

    raise HTTPException(status_code=404, detail="No model found with the provided ID.")
