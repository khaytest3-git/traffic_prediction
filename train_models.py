"""
Train LR and GB models on the full Chicago dataset and save to disk.

Run this locally before deploying:
    python train_models.py

Outputs:
    lr_model.joblib
    gb_model.joblib
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent

CANDIDATE_DATASETS = [
    BASE_DIR / "Chicago_Traffic_Tracker_-_Historical_Congestion_Estimates_by_Segment_-_2024-Current_20260222.csv",
    BASE_DIR / "traffic_sample.csv",
]

LR_MODEL_PATH = BASE_DIR / "lr_model.joblib"
GB_MODEL_PATH = BASE_DIR / "gb_model.joblib"


def load_dataset():
    for path in CANDIDATE_DATASETS:
        if path.exists():
            print(f"Loading dataset: {path.name} ({path.stat().st_size // 1_000_000} MB)")
            return pd.read_csv(path)
    raise FileNotFoundError("No dataset found. Place traffic_sample.csv in the project root.")


def prepare(df):
    df = df[['SPEED', 'HOUR', 'DAY_OF_WEEK']].dropna()
    df = df[df['SPEED'] > 0].copy()
    df['CONGESTION'] = np.where(df['SPEED'] < 20, 1, 0)
    return df


def main():
    df = prepare(load_dataset())

    print(f"Rows after cleaning: {len(df):,}")
    print(f"Congestion rate: {df['CONGESTION'].mean():.1%}\n")

    X = df[['HOUR', 'DAY_OF_WEEK']]
    y = df['CONGESTION']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training on {len(X_train):,} rows, testing on {len(X_test):,} rows\n")

    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train, y_train)
    print("Logistic Regression Results:")
    print(classification_report(y_test, lr.predict(X_test), zero_division=0))

    print("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    print("Gradient Boosting Results:")
    print(classification_report(y_test, gb.predict(X_test), zero_division=0))

    joblib.dump(lr, LR_MODEL_PATH)
    joblib.dump(gb, GB_MODEL_PATH)
    print(f"Saved: {LR_MODEL_PATH.name}")
    print(f"Saved: {GB_MODEL_PATH.name}")


if __name__ == "__main__":
    main()
