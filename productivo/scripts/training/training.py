"""Script para el training del modelo."""

import argparse
import glob
import json
import logging
import os
import pandas as pd
import pickle as pkl
import random
import xgboost


def parse_args():
    """Parse arguments with `argparse.ArgumentParser`."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_depth",        type=int,   default=3)     # Maximum depth of the trees
    parser.add_argument("--eta",              type=float, default=0.10)  # Learning rate
    parser.add_argument("--gamma",            type=float, default=0.00)  # Minimum loss reduction required to make a further partition
    parser.add_argument("--min_child_weight", type=int,   default=1)     # Minimum sum of the instance weight needed in a child

    parser.add_argument("--silent",           type=int,   default=0)                  # Antonym to verbosity
    parser.add_argument("--objective",        type=str,   default="binary:logistic")  # Learning objective
    parser.add_argument("--num_round",        type=int,   default=10)                 # Number of boosting rounds
    parser.add_argument("--eval_metric",      type=str,   default="auc")              # Evaluation metric

    parser.add_argument("--train",      type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))

    return parser.parse_args()


def main():
    """Main function."""
    # Parse args
    args = parse_args()

    print("Train contents:") 
    train_files = os.listdir(args.train)
    print(train_files)

    print("Val contents:")
    val_files = os.listdir(args.validation)
    print(val_files)

    train_path = os.path.join(args.train,      [f for f in train_files if f.endswith(".csv")][0])
    val_path   = os.path.join(args.validation, [f for f in val_files if f.endswith(".csv")][0])

    print("Loading train dataframe...")
    train_X = pd.read_csv(train_path)
    train_y = train_X[["RainTomorrow"]]
    train_X = train_X.drop("RainTomorrow", axis=1)

    print("Loading val dataframe...")
    val_X = pd.read_csv(val_path)
    val_y = val_X[["RainTomorrow"]]
    val_X = val_X.drop("RainTomorrow", axis=1)

    # Modify data format and type
    train_X = train_X.values
    train_y = train_y.values.reshape(-1)

    val_X = val_X.values
    val_y = val_y.values.reshape(-1)

    print(f"Train features shape: {train_X.shape}")
    print(f"Train labels shape:   {train_y.shape}")
    print(f"Val features shape:   {val_X.shape}")
    print(f"Val labels shape:     {val_y.shape}")

    dtrain = xgboost.DMatrix(train_X, label=train_y)
    dval   = xgboost.DMatrix(val_X, label=val_y)

    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "silent": args.silent,
        "objective": args.objective,
        "eval_metric": args.eval_metric
    }

    # Train the model
    bst = xgboost.train(
        params=params,
        dtrain=dtrain,
        evals=[(dtrain, "train"), (dval, "val")],
        num_boost_round=args.num_round,
    )

    # Evaluate the model
    eval_results = bst.eval(dval)
    validation_auc = eval_results["validation-auc"]

    # Log the metric
    logging.info(f"validation-auc:{validation_auc}")

    # Save model
    model_path = os.path.join(os.environ.get("SM_MODEL_DIR"), "model.bin")
    with open(model_path, "wb") as f:
        pkl.dump(bst, f)


if __name__ == "__main__":
    main()
