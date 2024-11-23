"""Script de inferencia para el modelo de XGBoost."""

import json
import numpy as np
import os
import pickle as pkl
import xgboost as xgb

from sagemaker_containers.beta.framework import content_types, worker
from sagemaker_xgboost_container import encoder as xgb_encoders


def input_fn(input_data: str, content_type):
    """Input function."""
    if content_type == content_types.JSON:
        obj = json.loads(input_data)
        features = obj["instances"][0]["features"]
        array = np.array(features).reshape((1, -1))
        return xgb.DMatrix(array)
    elif content_type == content_types.CSV:
        array = np.fromstring(input_data, sep=",").reshape((1, -1))
        return xgb.DMatrix(array)
    else:
        return xgb_encoders.decode(input_data, content_type)


def model_fn(model_dir):
    """Model function."""
    model_file = os.path.join(model_dir, "model.bin")
    with open(model_file, "rb") as f:
        model = pkl.load(f)
    return model


def threshold_score(score: float) -> str:
    return "yes" if score > 0.5 else "no"


def output_fn(prediction, accept: str) -> worker.Response:
    """Output function."""
    pred_array_value = np.array(prediction)
    score = pred_array_value[0]
    
    if accept == content_types.JSON:
        return_value = {
            "predictions": [
                {
                    "score": score.astype(float),
                    "predicted_label": threshold_score(score),
                }
            ]
        }
        return worker.Response(json.dumps(return_value), mimetype=accept)
    elif accept == content_types.CSV:
        return worker.Response(threshold_score(score), mimetype=accept)
    else:
        raise RuntimeError(f"{accept} accept type is not supported.")
