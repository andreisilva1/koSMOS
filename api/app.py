from contextlib import asynccontextmanager
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
from utils.global_cleaner import global_cleaner
from model_tests.classification import test_classification_algorithms
from utils.dataframes import (
    compact_file_to_less_than_max_size_mb,
)
from utils.extractors import extract_numericals_categoricals_and_ordinals
from utils.conversor import convert_to_df
from database.session import create_local_tables
from dependencies import ModelServiceDep
from checks.request import check_dict_values
from model_tests.regression import test_regression_algorithms
from database.models import MLModel
from model_tests.clustering import test_clustering_algorithms
import zipfile
from cryptography.fernet import Fernet

load_dotenv()

MAX_POSSIBLE_SIZE_ORIGINAL_FILE = (
    200 * 1024 * 1024
)  # If the csv is less than 200mb, will compact him to ~25MB, otherwise, no chance...
MAX_SIZE_TO_SAVE = 25 * 1024 * 1024  # 25MB is the maximum size of test datasets for now

ALLOW_LOCAL_FALLBACK = os.getenv("ALLOW_LOCAL_FALLBACK")


try:
    ALLOW_LOCAL_FALLBACK = bool(int(ALLOW_LOCAL_FALLBACK))
except:
    ALLOW_LOCAL_FALLBACK = False


@asynccontextmanager
async def lifespan_handler(app: FastAPI):
    if ALLOW_LOCAL_FALLBACK is True:
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
    id_columns: str = Form(None),
    target: str = Form(None),
    target_type: str = Form(None),
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

    if id_columns:
        df.drop(columns=[column for column in json.loads(id_columns)], inplace=True)

    clean_df = global_cleaner(df)

    valid_dict_types = {}
    any_str_columns = []
    for key, value in dict_types.items():
        if (
            value["values"] == "any" and value["col_type"] == "str"
        ):  # Remove values any-string from model testing
            any_str_columns.append(key)
        else:
            valid_dict_types[key] = value

    numericals, categoricals, ordinals = extract_numericals_categoricals_and_ordinals(
        valid_dict_types
    )

    if any_str_columns:
        clean_df.drop(
            columns=[col for col in any_str_columns if col in clean_df.columns],
            inplace=True,
        )

    if target:
        check_dict_values(dict_types, dict_values)
        # Check if target is numeric or categoric

        if pd.api.types.is_numeric_dtype(df[target]) and target_type == "numerical":
            (
                best_model,
                pp,
                stats_df,
                csv_high_correlations,
                csv_all_correlations,
            ) = test_regression_algorithms(
                target=target,
                df=clean_df,
                numericals=numericals,
                categoricals=categoricals,
                ordinals=ordinals,
            )

        else:
            (
                best_model,
                pp,
                stats_df,
                csv_high_correlations,
                csv_all_correlations,
            ) = test_classification_algorithms(
                target=target,
                df=clean_df,
                numericals=numericals,
                categoricals=categoricals,
                ordinals=ordinals,
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

        # Put the final dataset, the high correlation, the all correlation dataset and the model itself in a zip
        output = BytesIO()

        with zipfile.ZipFile(output, "w") as z:
            z.writestr("prediction.csv", prediction_df.to_csv(index=False)),
            z.writestr("stats_df.csv", stats_df.to_csv(index=False)),
            z.writestr("high_correlations.csv", csv_high_correlations)
            z.writestr("all_correlations.csv", csv_all_correlations)
            z.writestr(f"ml_model.pkl", ml_model)
            z.writestr(f"preprocessor.pkl", preprocessor_pkl)

        output.seek(0)
    if not target:
        if not n_groups:
            # No target AND n_groups? hierarchical cluster.
            cluster_method = "hierarchical"
        else:
            # No target WITH n_groups? K-means.
            cluster_method = "k-means"

        (
            stats_df,
            knn,
            csv_high_correlations,
            csv_all_correlations,
            best_model,
            pp,
        ) = test_clustering_algorithms(
            cluster_method=cluster_method,
            df=clean_df,
            n_groups=n_groups,
            numericals=numericals,
            ordinals=ordinals,
        )
        X_transformed = pp.fit_transform(df)

        ml_model = pickle.dumps(best_model)
        preprocessor = pickle.dumps(pp)

        best_model.fit_predict(X_transformed)
        df["cluster"] = best_model.labels_
        # Put the final dataset, the high correlation, the all correlation dataset and the model itself in a zip
        output = BytesIO()
        with zipfile.ZipFile(output, "w") as z:
            z.writestr("dataset.csv", df.to_csv(index=False))
            z.writestr("stats_df.csv", stats_df.to_csv(index=False)),
            z.writestr("high_correlations.csv", csv_high_correlations)
            z.writestr("all_correlations.csv", csv_all_correlations)
            if knn:
                z.writestr("knn.pkl", pickle.dumps(knn)),
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
    model_file: UploadFile,
    knn_file: UploadFile = Form(None),
    model_id: str = Form(None),
    dict_types: str = Form(None),
    target: str = Form(None),
):
    # The bytes of the model will be save in the database
    model_contents = await model_file.read()
    preprocessor_content = await preprocessor_file.read()
    knn_content = None
    if knn_file:
        knn_content = await knn_file.read()

    await service.save_model(
        MLModel(
            name=model_id,
            preprocessor=preprocessor_content,
            knn=knn_content,
            model=model_contents,
            dict_types=dict_types,
            target=target if target else None,
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

        load_model = joblib.load(BytesIO(model["model"]))
        check_dict_values(dict_types, dict_values)

        target = model.get("target", None)

        df_values = DataFrame([dict_values])

        if hasattr(load_model, "predict"):
            y_predict = load_model.predict(df_values)
        else:
            knn = joblib.load(BytesIO(model["knn"]))
            y_predict = knn.predict(df_values)

        y_predict = y_predict.item()
        if target:
            dict_values[target] = y_predict
        else:
            dict_values["cluster"] = y_predict
        return dict_values

    raise HTTPException(status_code=404, detail="No model found with the provided ID.")
