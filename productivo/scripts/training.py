"""Script para el training del modelo."""

import argparse
import glob
import json
import os
import pandas as pd
import pickle as pkl
import random
import xgboost


def parse_args():
    """Parse arguments with `argparse.ArgumentParser`."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_depth",        type=int,   default=5)
    parser.add_argument("--eta",              type=float, default=0.05)
    parser.add_argument("--gamma",            type=int,   default=4)
    parser.add_argument("--min_child_weight", type=int,   default=6)
    parser.add_argument("--silent",           type=int,   default=0)
    parser.add_argument("--objective",        type=str,   default="binary:logistic")
    parser.add_argument("--eval_metric",      type=str,   default="auc")
    parser.add_argument("--num_round",        type=int,   default=10)

    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--val",   type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    # train_files_path, val_files_path = args.train, args.val

    train_features_path = os.path.join(args.train, 'train_features.csv')
    train_labels_path   = os.path.join(args.train, 'train_labels.csv')
    
    val_features_path = os.path.join(args.val, 'val_features.csv')
    val_labels_path   = os.path.join(args.val, 'val_labels.csv')
    
    print("Loading train dataframes...")
    df_train_features = pd.read_csv(train_features_path, header=None)
    df_train_labels   = pd.read_csv(train_labels_path, header=None)
    
    print("Loading val dataframes...")
    df_val_features = pd.read_csv(val_features_path, header=None)
    df_val_labels   = pd.read_csv(val_labels_path, header=None)
    
    X = df_train_features.values
    y = df_train_labels.values.reshape(-1)

    val_X = df_val_features.values
    val_y = df_val_labels.values.reshape(-1)

    print(f"Train features shape: {X.shape}")
    print(f"Train labels shape:   {y.shape}")
    print(f"Val features shape:   {val_X.shape}")
    print(f"Val labels shape:     {val_y.shape}")

    dtrain = xgboost.DMatrix(X, label=y)
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

    bst = xgboost.train(
        params=params,
        dtrain=dtrain,
        evals=[(dtrain, "train"), (dval, "val")],
        num_boost_round=args.num_round,
    )
    
    model_dir = os.environ.get("SM_MODEL_DIR")
    model_path = os.path.join(model_dir, "model.bin")
    with open(model_path, "wb") as f:
        pkl.dump(bst, f)


if __name__ == "__main__":
    main()
