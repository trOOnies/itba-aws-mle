"""Script de inferencia."""

from __future__ import print_function

import csv
from io import StringIO
import json
import os
import numpy as np
import pandas as pd
import sys

from sklearn.externals import joblib
from sagemaker_containers.beta.framework import (
    content_types,
    encoders,
    env,
    modules,
    transformer,
    worker,
)

feature_columns_names = [
    # "Date",
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "WindGustSpeed",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Humidity9am",	
    "Humidity3pm",	
    "Pressure9am",	
    "Pressure3pm",
    "Temp9am",
    "Temp3pm",
    "RainToday",
    "WindGustDir_east",
    "WindGustDir_north",
    "WindDir9am_east",	
    "WindDir9am_north",	
    "WindDir3pm_east",
    "WindDir3pm_north",
]


def input_fn(input_data: str, content_type: str) -> pd.DataFrame:
    """Input function."""
    if content_type != "text/csv":
        raise ValueError(f"{content_type} not supported by script!")

    print(input_data)

    df = pd.read_csv(StringIO(input_data), header=None)
    if df.shape[1] == len(feature_columns_names):
        df.columns = feature_columns_names

    return df


def predict_fn(input_data, model):
    """Predict function."""
    features = model.transform(input_data)
    return features


def output_fn(prediction, accept):
    """Output function."""
    if accept == "application/json":
        json_output = {"instances": [{"features": row} for row in prediction.tolist()]}
        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException(f"{accept} accept type is not supported.")


def model_fn(model_dir):
    """Model function."""
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor
