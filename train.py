"""End-to-end: data/{train,test}.csv -> data/submission.csv with CV report."""
import os
import sys
import numpy as np
import pandas as pd

from src.features import build_features
from src.models import make_models, train_oof, blend
from src.evaluate import rmsle, results_table

DATA_DIR = "data"


def main():
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        sys.exit(f"ERROR: place train.csv and test.csv in ./{DATA_DIR}/ first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Loaded train={train_df.shape} test={test_df.shape}")

    X, y, X_test, feats = build_features(train_df, test_df)
    print(f"Built {len(feats)} features")

    scores, oof_dict, test_dict = {}, {}, {}
    for name in make_models().keys():
        print(f"Training {name} (5-fold OOF)...")
        oof, test_pred, cv = train_oof(name, X, y, X_test, folds=5)
        scores[name] = cv
        oof_dict[name] = oof
        test_dict[name] = test_pred
        print(f"  {name} OOF RMSLE = {cv:.5f}")

    blended_oof, blended_test, weights = blend(oof_dict, y, test_dict)
    scores["ensemble"] = rmsle(np.expm1(y.values), np.expm1(blended_oof))
    print(f"Blend weights: {weights}")
    print("\n" + results_table(scores))

    preds = np.clip(np.expm1(blended_test), 0, None)
    out = pd.DataFrame({"id": test_df["id"].astype(int), "revenue": preds})
    out_path = os.path.join(DATA_DIR, "submission.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
